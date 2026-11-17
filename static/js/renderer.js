// Time Cube renderer (WebGL2).
//
// The last N source frames live in one 3D texture used as a ring buffer: layer `head` is the newest frame.
// Depth d (0 = front face, 1 = back face) maps to a frame age, so the cube is the video's recent history:
//   Wire   - each of P planes is a triangulated grid drawn as anti-aliased lines in the colours of its frame;
//   Slices - P see-through planes, drawn back to front (grids, when they have relief);
//   Solid  - the box faces sample the volume where they are: the front is the newest frame, the sides are the
//            edge pixels of every frame (a slit-scan);
//   Points - P frames as clouds of round points.
// Each texel's alpha holds that frame's AI depth (1 = nearest), when the video has it. Relief pushes near things
// toward the front face, as the camera saw them; without AI depth, dark parts count as near.
// The shape of time bends the block: Stack leans the older frames up, Echo shrinks them toward the middle, and
// Spiral turns them around the time axis. Every look places its points through shape(), so they all follow.
// The Camera path layout places every frame where its camera was instead: each layer keeps that camera's pose (a
// 4 x N float texture next to the volume), and a frame's points go out from it to their distance (the AI depth).
// The viewer then sits at the newest camera, so the front view is the video itself.
// A followed object: a second volume holds its mask in every frame (Isolate tracked / Dim the rest), and each
// layer keeps the object's centre for the track line.
// The scene is drawn into a transparent multisampled buffer, so the glow can be made from the cube alone and laid
// over the background colour in the last pass.
import { lookWeights } from "./camera.js";
import { identity, invertRigid, multiply, perspective, rotation, transformPoint, translate, unrotate } from "./mat.js";
import { DEPTH_C, distance } from "./paths.js";

const VOL_GLSL = `
precision highp sampler3D;
uniform sampler3D uVol;
uniform float uHead;    // newest layer
uniform float uLayers;  // N
uniform float uFlip;    // 1: the newest frame is at the back
uniform vec2 uVolTexel; // 1 / layer size
uniform float uMirror, uSoften, uToneB, uToneC, uToneS;
uniform float uAi;      // 1: nearness from the AI depth (the volume's alpha); 0: from brightness (dark = near)
uniform float uRelief, uTrimFar, uTrimDark, uMotion, uFadeOld;
uniform sampler3D uMaskVol;   // the followed object in every frame (1 = the object)
uniform float uTrackDim;      // how much of the rest fades away (1: only the object stays)
uniform vec3 uDimTo;          // ...into the background colour (fading alone would pile up again over the frames)
float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
// The frame at depth d (0 = front face, 1 = back face), with the Source tone applied (so tone changes reach the
// whole history at once). .a is how near that point was to the camera: 0 far (the sky) .. 1 nearest.
vec4 vol(vec2 uv, float d) {
  if (uMirror > 0.5) uv.x = 1.0 - uv.x;
  float dd = mix(d, 1.0 - d, uFlip);
  float r = (uHead - dd * (uLayers - 1.0) + 0.5) / uLayers;
  vec4 t = textureLod(uVol, vec3(uv, r), 0.0);
  vec3 c = t.rgb;
  if (uSoften > 0.001) {
    vec2 o = uSoften * 1.5 * uVolTexel;
    c = c * 0.36 + (textureLod(uVol, vec3(uv + vec2(o.x, o.y), r), 0.0).rgb +
                    textureLod(uVol, vec3(uv + vec2(-o.x, o.y), r), 0.0).rgb +
                    textureLod(uVol, vec3(uv + vec2(o.x, -o.y), r), 0.0).rgb +
                    textureLod(uVol, vec3(uv - o, r), 0.0).rgb) * 0.16;
  }
  c = (c - 0.5) * uToneC + 0.5 + uToneB;
  c = clamp(mix(vec3(luma(c)), c, uToneS), 0.0, 1.0);
  return vec4(c, mix(1.0 - luma(c), t.a, uAi));
}
// Nearness only, from one texel (no softening or tone): for quick tests around a point.
float nearQuick(vec2 uv, float d) {
  if (uMirror > 0.5) uv.x = 1.0 - uv.x;
  float dd = mix(d, 1.0 - d, uFlip);
  vec4 t = textureLod(uVol, vec3(uv, (uHead - dd * (uLayers - 1.0) + 0.5) / uLayers), 0.0);
  return mix(1.0 - luma(t.rgb), t.a, uAi);
}
// Relief: near things come toward the front face (the way the camera looked), far things go back.
vec3 lift(float near) { return vec3(0.0, 0.0, uRelief * (near - 0.5) * 0.5); }
// How much of a point is left after the clean-up: far away (the sky) and dark pixels trimmed.
float keep(vec4 v) {
  float k = 1.0;
  if (uTrimFar > 0.001) k *= smoothstep(uTrimFar - 0.04, uTrimFar + 0.04, v.a);
  if (uTrimDark > 0.001) k *= smoothstep(uTrimDark - 0.04, uTrimDark + 0.04, luma(v.rgb));
  return k;
}
// The followed object: 1 on it, and toward 0 elsewhere as the rest is dimmed.
float tracked(vec2 uv, float d) {
  if (uTrackDim < 0.001) return 1.0;
  if (uMirror > 0.5) uv.x = 1.0 - uv.x;
  float dd = mix(d, 1.0 - d, uFlip);
  float m = textureLod(uMaskVol, vec3(uv, (uHead - dd * (uLayers - 1.0) + 0.5) / uLayers), 0.0).r;
  return mix(1.0, smoothstep(0.25, 0.75, m), uTrackDim);
}
// Fade older frames: the newest frame stays whole, the older ones thin out (1: a short ghost trail).
float ageFade(float d) {
  float age = clamp(mix(d, 1.0 - d, uFlip), 0.0, 1.0);
  return pow(1.0 - age, 6.0 * uFadeOld);
}
// Highlight motion: what looks the same as one frame earlier fades away, what changed stays.
float moving(vec2 uv, float d, vec3 c) {
  if (uMotion < 0.001) return 1.0;
  float s = 1.0 / max(uLayers - 1.0, 1.0);
  vec3 o = vol(uv, d + s <= 1.0 ? d + s : d - s).rgb;
  return mix(1.0, smoothstep(0.035, 0.16, length(c - o)), uMotion);
}
`;

// The Camera path layout: where a frame's points go.
const PATH_GLSL = `
uniform int uLayout;               // 0: the time cube, 1: along the camera path
uniform highp sampler2D uPoses;    // the camera of every layer: row = layer, 4 texels = its matrix's columns
uniform vec2 uTanHalf;             // the camera's tan(half its field of view): across, down
uniform float uDepthC, uPivotZ;    // nearness -> distance rule; the distance frames sit at with no relief
// The layer that holds the frame at depth d, and d moved onto it, so a frame's colours are never mixed with the
// next frame's (they were seen from another place).
float layerAt(float d) {
  float dd = mix(d, 1.0 - d, uFlip);
  return mod(floor(uHead - dd * (uLayers - 1.0) + 0.5), uLayers);
}
float snapD(float d) {
  float dd = mix(d, 1.0 - d, uFlip);
  dd = floor(dd * (uLayers - 1.0) + 0.5) / max(uLayers - 1.0, 1.0);
  return mix(dd, 1.0 - dd, uFlip);
}
mat4 poseAt(float d) {
  int y = int(layerAt(d));
  return mat4(texelFetch(uPoses, ivec2(0, y), 0), texelFetch(uPoses, ivec2(1, y), 0),
              texelFetch(uPoses, ivec2(2, y), 0), texelFetch(uPoses, ivec2(3, y), 0));
}
// Relief 1 is the true distance, 0 puts the whole frame at the pivot distance (a flat picture on the path).
float camDist(float near) { return mix(uPivotZ, 1.0 / (uDepthC + (1.0 - uDepthC) * near), uRelief); }
// How whole the surface is around uv: 0 where the distance jumps (the edge between something near and something
// far), so a sheet or a line is cut there instead of being stretched across the gap.
float wholeAt(float za, float zb) { return 1.0 - smoothstep(1.12, 1.35, max(za, zb) / max(min(za, zb), 1e-4)); }
float whole(vec2 uv, float d, vec2 step) {
  float z = camDist(nearQuick(uv, d)), lo = z, hi = z;
  for (int i = 0; i < 4; i++) {
    vec2 o = i == 0 ? vec2(step.x, 0.0) : i == 1 ? vec2(-step.x, 0.0) : i == 2 ? vec2(0.0, step.y) : vec2(0.0, -step.y);
    float zi = camDist(nearQuick(clamp(uv + o, 0.0, 1.0), d));
    lo = min(lo, zi);
    hi = max(hi, zi);
  }
  return wholeAt(lo, hi);
}
vec3 alongPath(vec2 uv, float d, float near) {
  float z = camDist(near);
  vec3 c = vec3((uv.x * 2.0 - 1.0) * uTanHalf.x * z, (1.0 - uv.y * 2.0) * uTanHalf.y * z, -z);
  return (poseAt(d) * vec4(c, 1.0)).xyz;
}
`;

const ECHO_BACK = 0.3;   // echo: the back face's size, next to the front's

// Where a point of the cube goes in the shape of time. p: position in the cube, d: 0 front face .. 1 back face.
const SHAPE_GLSL = `
uniform int uShape;       // 0 cube, 1 stack, 2 echo
uniform float uSpiral;    // turn of the back face, radians
uniform float uRise;      // stack: how far the back face is lifted
vec3 shape(vec3 p, float d) {
  if (uShape == 1) p.y += d * uRise;
  else if (uShape == 2) p.xy *= mix(1.0, ${ECHO_BACK.toFixed(3)}, d);
  if (uSpiral != 0.0) {
    float a = uSpiral * d, c = cos(a), s = sin(a);
    p.xy = vec2(c * p.x - s * p.y, s * p.x + c * p.y);
  }
  return p;
}
`;

const QUAD_VS = `#version 300 es
out vec2 vUv;
void main() {
  vec2 p = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
  vUv = p * 0.5 + 0.5;
  gl_Position = vec4(p, 0.0, 1.0);
}`;

// Source frame -> one layer of the volume, cropped to the volume's shape (cover fit). Alpha: its AI depth.
const WRITE_FS = `#version 300 es
precision highp float;
uniform sampler2D uSrc, uDepth, uMask;
uniform int uHasDepth;     // 0: this frame has no AI depth (alpha 0.5, flat)
uniform int uHasMask;      // 0: no followed object in this frame
uniform vec2 uSize;        // volume layer size in texels
uniform vec4 uCrop;        // source uv = crop.xy + uv * crop.zw
layout(location = 0) out vec4 o;
layout(location = 1) out vec4 oMask;
void main() {
  vec2 uv = uCrop.xy + gl_FragCoord.xy / uSize * uCrop.zw;   // row 0 of the layer = top row of the picture
  o = vec4(texture(uSrc, uv).rgb, uHasDepth == 1 ? texture(uDepth, uv).r : 0.5);
  oMask = vec4(uHasMask == 1 ? texture(uMask, uv).r : 0.0, 0.0, 0.0, 1.0);
}`;

// The block's sides are strips along the depth, so they can lean, taper and twist with the shape of time.
// vP stays the plain cube position: that is where the volume is sampled.
const SOLID_VS = `#version 300 es
${SHAPE_GLSL}
in vec3 aPos;      // unit cube point, -1..1
in vec3 aNrm;
uniform mat4 uMVP, uModel;
uniform vec3 uHalf;
out vec3 vP;
out float vFacing;
void main() {
  vP = aPos;
  vFacing = abs((uModel * vec4(aNrm, 0.0)).z);
  gl_Position = uMVP * vec4(shape(aPos * uHalf, 0.5 - aPos.z * 0.5), 1.0);
}`;

const SOLID_FS = `#version 300 es
precision highp float;
${VOL_GLSL}
in vec3 vP;
in float vFacing;
uniform float uAlpha, uShade;
out vec4 o;
void main() {
  vec3 p = clamp(vP, -1.0, 1.0);
  vec4 v = vol(vec2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5), 0.5 - p.z * 0.5);
  float f = tracked(vec2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5), 0.5 - p.z * 0.5);
  vec3 c = mix(uDimTo, v.rgb * (1.0 - uShade * (1.0 - vFacing)), f);
  float a = uAlpha * keep(v) * f;
  o = vec4(c * a, a);
}`;

// A slice is a grid of uGrid cells (1 x 1 when flat), so it can take the relief.
const SLICE_VS = `#version 300 es
precision highp float;
precision highp int;
${VOL_GLSL}
${SHAPE_GLSL}
${PATH_GLSL}
uniform mat4 uMVP;
uniform vec3 uHalf;
uniform int uPlanes;
uniform int uReverse;   // 1: draw from the back plane to the front plane
uniform ivec2 uGrid;
out vec2 vUv;
out float vD;
out float vWhole;
void main() {
  int k = uReverse == 1 ? gl_InstanceID : uPlanes - 1 - gl_InstanceID;
  float d = 1.0 - float(k) / float(max(uPlanes - 1, 1));
  if (uLayout == 1) d = snapD(d);
  int cell = gl_VertexID / 6, v = gl_VertexID - cell * 6;
  int cy = cell / uGrid.x, cx = cell - cy * uGrid.x;
  vec2 c = vec2((v == 1 || v == 2 || v == 4) ? 1.0 : 0.0, (v == 2 || v == 4 || v == 5) ? 1.0 : 0.0);
  c = (vec2(cx, cy) + c) / vec2(uGrid);
  vUv = vec2(c.x, 1.0 - c.y);
  vD = d;
  vec3 pos;
  vWhole = 1.0;
  if (uLayout == 1) {
    pos = alongPath(vUv, d, vol(vUv, d).a);
    vWhole = whole(vUv, d, 1.0 / vec2(uGrid));
  } else {
    pos = vec3((c.x * 2.0 - 1.0) * uHalf.x, (c.y * 2.0 - 1.0) * uHalf.y, uHalf.z * (1.0 - 2.0 * d));
    if (uRelief != 0.0) pos += lift(vol(vUv, d).a);
    pos = shape(pos, d);
  }
  gl_Position = uMVP * vec4(pos, 1.0);
}`;

const SLICE_FS = `#version 300 es
precision highp float;
${VOL_GLSL}
in vec2 vUv;
in float vD;
in float vWhole;
uniform float uAlpha, uClear;
out vec4 o;
void main() {
  vec4 v = vol(vUv, vD);
  float f = tracked(vUv, vD);
  float a = uAlpha * (1.0 - uClear * smoothstep(0.35, 0.95, luma(v.rgb))) * keep(v) * ageFade(vD) * vWhole * f;
  v.rgb = mix(uDimTo, v.rgb, f);
  if (a > 0.002) a *= moving(vUv, vD, v.rgb);
  o = vec4(v.rgb * a, a);
}`;

// Points: P frames, each a uGrid of round points in its colours, lifted by the relief. Drawn back to front.
// A point's size follows the grid spacing on screen (uSize 1 = touching its neighbours), so zooming and the output
// size keep the cloud as dense.
const POINT_VS = `#version 300 es
precision highp float;
precision highp int;
${VOL_GLSL}
${SHAPE_GLSL}
${PATH_GLSL}
uniform mat4 uMVP;
uniform vec3 uHalf;
uniform int uPlanes, uReverse;
uniform ivec2 uGrid;
uniform float uSize, uFocal, uAlpha;
out vec4 vColor;
out float vPx;
void main() {
  int per = uGrid.x * uGrid.y;
  int i = gl_VertexID / per, j = gl_VertexID - i * per;
  int k = uReverse == 1 ? i : uPlanes - 1 - i;
  float d = 1.0 - float(k) / float(max(uPlanes - 1, 1));
  if (uLayout == 1) d = snapD(d);
  int gy = j / uGrid.x, gx = j - gy * uGrid.x;
  vec2 uv = (vec2(gx, gy) + 0.5) / vec2(uGrid);
  vec4 v = vol(uv, d);
  float f = tracked(uv, d);
  float a = uAlpha * keep(v) * ageFade(d) * f;
  if (a > 0.004) a *= moving(uv, d, v.rgb);
  vec3 pos;
  float cell;   // the grid's spacing where the point is
  if (uLayout == 1) {
    pos = alongPath(uv, d, v.a);
    cell = 2.0 * uTanHalf.x * camDist(v.a) / float(uGrid.x);
  } else {
    pos = shape(vec3((uv.x * 2.0 - 1.0) * uHalf.x, (1.0 - uv.y * 2.0) * uHalf.y, uHalf.z * (1.0 - 2.0 * d)) + lift(v.a), d);
    cell = (2.0 * uHalf.x / float(uGrid.x)) * (uShape == 2 ? mix(1.0, ${ECHO_BACK.toFixed(3)}, d) : 1.0);
  }
  gl_Position = uMVP * vec4(pos, 1.0);
  float px = cell * uFocal / max(gl_Position.w, 1e-4) * uSize;
  vPx = clamp(px, 1.0, 64.0);
  gl_PointSize = vPx;
  if (a <= 0.004) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);   // nothing to draw: off the screen
  vColor = vec4(mix(uDimTo, v.rgb, f), a);
}`;

const POINT_FS = `#version 300 es
precision highp float;
in vec4 vColor;
in float vPx;
out vec4 o;
void main() {
  float r = length(gl_PointCoord * 2.0 - 1.0);
  float edge = vPx < 2.5 ? 1.0 : 1.0 - smoothstep(1.0 - 2.0 / vPx, 1.0, r);   // round, with a soft rim
  if (edge <= 0.0) discard;
  float a = vColor.a * edge;
  o = vec4(vColor.rgb * a, a);
}`;

// Wire: every instance is one grid segment of one plane, expanded to a screen-space quad of uWidth pixels.
// A plane is a triangulated grid with in-plane coordinates (a, b) at position p along its axis; uAxes maps
// (a, b, p) to the volume's (u, v, d). Columns planes (constant u) run through time, like the reel's long lines.
// Segments are numbered: edges along a, then edges along b, then one diagonal per cell.
const WIRE_VS = `#version 300 es
precision highp float;
precision highp int;
${VOL_GLSL}
${SHAPE_GLSL}
${PATH_GLSL}
uniform mat4 uMVP;
uniform vec3 uHalf;
uniform mat3 uAxes;
uniform vec3 uNormal;   // the plane axis in cube units, for the depth relief
uniform ivec2 uGrid;
uniform int uSegs, uPlanes;
uniform float uWidth, uDisp;
uniform vec2 uViewport;
out vec4 vColor;
out float vAcross;
vec3 place(vec3 uvd, vec4 v) {
  if (uLayout == 1) return alongPath(uvd.xy, uvd.z, v.a);
  vec3 p = vec3((uvd.x * 2.0 - 1.0) * uHalf.x, (1.0 - uvd.y * 2.0) * uHalf.y, uHalf.z * (1.0 - 2.0 * uvd.z));
  return shape(p + uNormal * (uDisp * (0.5 - luma(v.rgb)) * 0.5) + lift(v.a), uvd.z);
}
void main() {
  int plane = gl_InstanceID / uSegs;
  int seg = gl_InstanceID - plane * uSegs;
  int ga = uGrid.x, gb = uGrid.y;
  int na = ga * (gb + 1), nb = (ga + 1) * gb;
  ivec2 a, b;
  if (seg < na) {
    int r = seg / ga; int c = seg - r * ga; a = ivec2(c, r); b = ivec2(c + 1, r);
  } else if (seg < na + nb) {
    int s = seg - na; int r = s / (ga + 1); int c = s - r * (ga + 1); a = ivec2(c, r); b = ivec2(c, r + 1);
  } else {
    int s = seg - na - nb; int r = s / ga; int c = s - r * ga; a = ivec2(c, r); b = ivec2(c + 1, r + 1);
  }
  float p = float(plane) / float(max(uPlanes - 1, 1));
  vec3 uvdA = uAxes * vec3(vec2(a) / vec2(uGrid), p);
  vec3 uvdB = uAxes * vec3(vec2(b) / vec2(uGrid), p);
  if (uLayout == 1) { uvdA.z = snapD(uvdA.z); uvdB.z = snapD(uvdB.z); }
  vec4 cA = vol(uvdA.xy, uvdA.z), cB = vol(uvdB.xy, uvdB.z);
  vec4 A = uMVP * vec4(place(uvdA, cA), 1.0);
  vec4 B = uMVP * vec4(place(uvdB, cB), 1.0);
  float fA = tracked(uvdA.xy, uvdA.z), fB = tracked(uvdB.xy, uvdB.z);
  float kA = keep(cA) * ageFade(uvdA.z) * fA, kB = keep(cB) * ageFade(uvdB.z) * fB;
  cA.rgb = mix(uDimTo, cA.rgb, fA);
  cB.rgb = mix(uDimTo, cB.rgb, fB);
  if (uLayout == 1) { float wAB = wholeAt(camDist(cA.a), camDist(cB.a)); kA *= wAB; kB *= wAB; }
  if (uMotion > 0.001) { kA *= moving(uvdA.xy, uvdA.z, cA.rgb); kB *= moving(uvdB.xy, uvdB.z, cB.rgb); }
  // a line reaching behind the viewer (the camera path can put frames there) is cut at the viewer
  if (A.w < 1e-3 && B.w < 1e-3) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); vColor = vec4(0.0); vAcross = 0.0; return; }
  if (A.w < 1e-3) A = mix(A, B, (1e-3 - A.w) / (B.w - A.w));
  if (B.w < 1e-3) B = mix(B, A, (1e-3 - B.w) / (A.w - B.w));
  vec2 half_ = uViewport * 0.5;
  vec2 sA = A.xy / A.w * half_, sB = B.xy / B.w * half_;
  vec2 dir = sB - sA;
  float len = length(dir);
  dir = len > 1e-4 ? dir / len : vec2(1.0, 0.0);
  vec2 nrm = vec2(-dir.y, dir.x);
  int v = gl_VertexID;
  float t = (v == 1 || v == 2 || v == 4) ? 1.0 : 0.0;
  float s = (v == 2 || v == 4 || v == 5) ? 1.0 : -1.0;
  float hw = max(uWidth, 1.0) * 0.5 + 1.0;
  vec4 P = mix(A, B, t);
  vec2 off = (nrm * s * hw + dir * (t * 2.0 - 1.0) * 0.5) / half_ * P.w;
  gl_Position = vec4(P.xy + off, P.zw);
  vColor = vec4(mix(cA.rgb, cB.rgb, t), mix(kA, kB, t));   // alpha: what the clean-up leaves of the line
  vAcross = s * hw;
}`;

const WIRE_FS = `#version 300 es
precision highp float;
in vec4 vColor;
in float vAcross;
uniform float uWidth, uAlpha, uInkAmt, uContrast;
uniform vec3 uInk;
out vec4 o;
void main() {
  float w = max(uWidth, 1.0);
  float cover = clamp(w * 0.5 + 0.5 - abs(vAcross), 0.0, 1.0) * min(uWidth, 1.0);
  vec3 c = clamp((vColor.rgb - 0.5) * uContrast + 0.5, 0.0, 1.0);
  c = mix(c, uInk, uInkAmt);
  float a = cover * uAlpha * vColor.a;
  o = vec4(c * a, a);
}`;

// Plain lines (the camera path and the outline of now): one segment per instance, expanded on screen like the wire.
const LINE_VS = `#version 300 es
precision highp float;
in vec3 aA, aB;
uniform mat4 uMVP;
uniform float uWidth;
uniform vec2 uViewport;
out float vAcross;
void main() {
  vec4 A = uMVP * vec4(aA, 1.0), B = uMVP * vec4(aB, 1.0);
  if (A.w < 1e-3 && B.w < 1e-3) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); vAcross = 0.0; return; }
  if (A.w < 1e-3) A = mix(A, B, (1e-3 - A.w) / (B.w - A.w));
  if (B.w < 1e-3) B = mix(B, A, (1e-3 - B.w) / (A.w - B.w));
  vec2 half_ = uViewport * 0.5;
  vec2 sA = A.xy / A.w * half_, sB = B.xy / B.w * half_;
  vec2 dir = sB - sA;
  float len = length(dir);
  dir = len > 1e-4 ? dir / len : vec2(1.0, 0.0);
  vec2 nrm = vec2(-dir.y, dir.x);
  int v = gl_VertexID;
  float t = (v == 1 || v == 2 || v == 4) ? 1.0 : 0.0;
  float s = (v == 2 || v == 4 || v == 5) ? 1.0 : -1.0;
  float hw = max(uWidth, 1.0) * 0.5 + 1.0;
  vec4 P = mix(A, B, t);
  gl_Position = vec4(P.xy + (nrm * s * hw + dir * (t * 2.0 - 1.0) * 0.5) / half_ * P.w, P.zw);
  vAcross = s * hw;
}`;

const LINE_FS = `#version 300 es
precision highp float;
in float vAcross;
uniform float uWidth;
uniform vec4 uColor;
out vec4 o;
void main() {
  float cover = clamp(max(uWidth, 1.0) * 0.5 + 0.5 - abs(vAcross), 0.0, 1.0) * min(uWidth, 1.0);
  float a = cover * uColor.a;
  o = vec4(uColor.rgb * a, a);
}`;

// Bright parts of the cube, at a quarter of the size: the start of the glow.
const BRIGHT_FS = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uTex;
uniform vec2 uTexel;
uniform float uThr;
out vec4 o;
void main() {
  vec4 c = (texture(uTex, vUv + uTexel * vec2(-1.0, -1.0)) + texture(uTex, vUv + uTexel * vec2(1.0, -1.0)) +
            texture(uTex, vUv + uTexel * vec2(-1.0, 1.0)) + texture(uTex, vUv + uTexel * vec2(1.0, 1.0))) * 0.25;
  vec3 straight = c.rgb / max(c.a, 1e-3);
  float l = dot(straight, vec3(0.2126, 0.7152, 0.0722));
  float k = smoothstep(uThr, min(uThr + 0.3, 1.0), l);
  o = vec4(c.rgb * k, c.a * k);
}`;

const BLUR_FS = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uTex;
uniform vec2 uStep;
out vec4 o;
void main() {
  vec4 s = texture(uTex, vUv) * 0.1964825501511404;
  s += (texture(uTex, vUv + uStep * 1.411764705882353) + texture(uTex, vUv - uStep * 1.411764705882353)) * 0.2969069646728344;
  s += (texture(uTex, vUv + uStep * 3.2941176470588234) + texture(uTex, vUv - uStep * 3.2941176470588234)) * 0.09447039785044732;
  s += (texture(uTex, vUv + uStep * 5.176470588235294) + texture(uTex, vUv - uStep * 5.176470588235294)) * 0.010381362401148057;
  o = s;
}`;

// Last pass. uOut 0: over the background colour; 1: over a checkerboard (to preview transparency);
// 2: transparent, straight (not premultiplied) alpha, for transparent PNG / ProRes 4444 / VP9 alpha.
const COMPOSE_FS = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uScene, uGlow, uCap;
uniform int uHasCap;       // 1: a caption over the picture (premultiplied, top row first)
uniform vec3 uBg;
uniform float uGlowAmt, uGrain, uSeed, uCheck;
uniform int uOut;
out vec4 o;
float hash(vec2 p) { p = fract(p * vec2(443.897, 441.423)); p += dot(p, p.yx + 19.19); return fract((p.x + p.y) * p.x); }
void main() {
  vec4 c = texture(uScene, vUv);
  vec4 g = texture(uGlow, vUv);
  vec3 glw = clamp(g.rgb * uGlowAmt * (1.0 - 0.65 * c.a), 0.0, 1.0);   // mostly around the cube, not over it
  float n = uGrain > 0.0 ? (hash(gl_FragCoord.xy + uSeed * 37.0) - 0.5) * uGrain * 0.12 : 0.0;
  if (uOut == 2) {
    // The same picture as a layer: over any background B it gives B(1-a)(1-g) + C(1-g) + g, which is what the
    // screen blend below gives over the background colour.
    float gm = max(glw.r, max(glw.g, glw.b));
    float a = 1.0 - (1.0 - c.a) * (1.0 - gm);
    vec3 p = c.rgb * (1.0 - gm) + glw;
    if (uHasCap == 1) {
      vec4 cp = texture(uCap, vUv);
      p = cp.rgb + p * (1.0 - cp.a);
      a = cp.a + a * (1.0 - cp.a);
    }
    vec3 straight = a > 1e-4 ? p / a : vec3(0.0);
    o = vec4(clamp(straight + n, 0.0, 1.0), a);
    return;
  }
  vec3 bg = uBg;
  if (uOut == 1) {
    vec2 q = floor(gl_FragCoord.xy / uCheck);
    bg = mod(q.x + q.y, 2.0) < 1.0 ? vec3(0.30) : vec3(0.40);
  }
  vec3 col = bg * (1.0 - c.a) + c.rgb;
  col = 1.0 - (1.0 - col) * (1.0 - glw);                               // screen: brightens, never darkens
  if (uHasCap == 1) {
    vec4 cp = texture(uCap, vUv);
    col = cp.rgb + col * (1.0 - cp.a);
  }
  o = vec4(clamp(col + n, 0.0, 1.0), 1.0);
}`;

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error("shader: " + log);
  }
  return sh;
}

function program(gl, vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error("program: " + gl.getProgramInfoLog(p));
  const u = {};
  const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) {
    const info = gl.getActiveUniform(p, i);
    u[info.name.replace(/\[0\]$/, "")] = gl.getUniformLocation(p, info.name);
  }
  return { p, u };
}

// The 6 faces of the -1..1 cube, wound counter-clockwise seen from outside (so back faces can be culled). The four
// sides are cut into `strips` along the depth (z), so the shape of time can bend them.
export const SOLID_STRIPS = 64;
function cubeMesh(strips = SOLID_STRIPS) {
  const faces = [
    [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
    [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
    [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
    [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
    [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
  ];
  const pos = [], nrm = [];
  for (const [n, u, v] of faces) {
    const corner = (a, b) => [0, 1, 2].map((i) => n[i] + a * u[i] + b * v[i]);
    const zAlongU = u[2] !== 0, zAlongV = v[2] !== 0;   // a side: one of its axes runs through time
    const nu = zAlongU ? strips : 1, nv = zAlongV ? strips : 1;
    for (let i = 0; i < nu; i++) {
      for (let j = 0; j < nv; j++) {
        const a0 = -1 + (2 * i) / nu, a1 = -1 + (2 * (i + 1)) / nu, b0 = -1 + (2 * j) / nv, b1 = -1 + (2 * (j + 1)) / nv;
        const q = [corner(a0, b0), corner(a1, b0), corner(a1, b1), corner(a0, b0), corner(a1, b1), corner(a0, b1)];
        for (const c of q) { pos.push(...c); nrm.push(...n); }
      }
    }
  }
  return { pos: new Float32Array(pos), nrm: new Float32Array(nrm), count: pos.length / 3 };
}

export const MAX_GRID = 128;
export const MAX_PLANES = 128;
export const MAX_POINT_GRID = 320;
export const SHAPES = ["cube", "stack", "echo"];

// Stack: the back face is lifted by the front face's height, so the older frames show above the newer ones.
const riseOf = (half) => 2 * half[1];

// shape() of the shaders, for one point (the block's centre and pivot).
export function shapePoint(s, half, p, d) {
  let [x, y, z] = p;
  if (s.shape === "stack") y += d * riseOf(half);
  else if (s.shape === "echo") { const k = 1 + (ECHO_BACK - 1) * d; x *= k; y *= k; }
  const a = (((s.spiral || 0) * Math.PI) / 180) * d;
  if (a) [x, y] = [Math.cos(a) * x - Math.sin(a) * y, Math.sin(a) * x + Math.cos(a) * y];
  return [x, y, z];
}
const SLICE_CELLS = 128;   // a slice with relief: cells along its long side

// The grid of one wire plane. axis: "columns" (planes across the width, running through time), "frames" (one
// plane per moment) or "rows" (horizontal planes through time). `grid` cells span the front face's long side;
// along the depth (time) a cell is `stretch` times longer.
export function wireGrid(w, half) {
  const cell = (2 * Math.max(half[0], half[1])) / Math.max(1, w.grid);
  const n = (ext, k = 1) => Math.max(1, Math.min(MAX_GRID, Math.round(ext / (cell * k))));
  const st = Math.max(1, w.stretch || 1);
  const [W, H] = [2 * half[0], 2 * half[1]];
  const D = (2 * half[2]) / st;
  if (w.axis === "frames") {
    // a -> u, b -> v, p -> d (mat3 columns)
    return { ga: n(W), gb: n(H), axes: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), normal: [0, 0, half[2]] };
  }
  if (w.axis === "rows") {
    // a -> d, b -> u, p -> v
    return { ga: n(D), gb: n(W), axes: new Float32Array([0, 0, 1, 1, 0, 0, 0, 1, 0]), normal: [0, half[1], 0] };
  }
  // columns: a -> d, b -> v, p -> u
  return { ga: n(D), gb: n(H), axes: new Float32Array([0, 0, 1, 0, 1, 0, 1, 0, 0]), normal: [half[0], 0, 0] };
}

export class Renderer {
  constructor(canvas) {
    const gl = canvas.getContext("webgl2", { alpha: false, antialias: false, depth: false, stencil: false,
      premultipliedAlpha: false, preserveDrawingBuffer: true, powerPreference: "high-performance" });
    if (!gl) throw new Error("WebGL2 is not available in this browser");
    this.gl = gl;
    this.canvas = canvas;
    this.samples = Math.min(4, gl.getParameter(gl.MAX_SAMPLES));
    this.progs = {
      write: program(gl, QUAD_VS, WRITE_FS),
      solid: program(gl, SOLID_VS, SOLID_FS),
      slice: program(gl, SLICE_VS, SLICE_FS),
      wire: program(gl, WIRE_VS, WIRE_FS),
      points: program(gl, POINT_VS, POINT_FS),
      line: program(gl, LINE_VS, LINE_FS),
      bright: program(gl, QUAD_VS, BRIGHT_FS),
      blur: program(gl, QUAD_VS, BLUR_FS),
      compose: program(gl, QUAD_VS, COMPOSE_FS),
    };
    this.emptyVao = gl.createVertexArray();
    this.cubeVao = gl.createVertexArray();
    gl.bindVertexArray(this.cubeVao);
    const mesh = cubeMesh();
    this.cubeCount = mesh.count;
    for (const [name, data] of [["aPos", mesh.pos], ["aNrm", mesh.nrm]]) {
      const loc = gl.getAttribLocation(this.progs.solid.p, name);
      const buf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      if (loc >= 0) {
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
      }
    }
    gl.bindVertexArray(null);
    // the path lines: two vec3 per instance
    this.lineVao = gl.createVertexArray();
    this.lineBuf = gl.createBuffer();
    gl.bindVertexArray(this.lineVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuf);
    for (const [name, off] of [["aA", 0], ["aB", 12]]) {
      const loc = gl.getAttribLocation(this.progs.line.p, name);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 24, off);
      gl.vertexAttribDivisor(loc, 1);
    }
    gl.bindVertexArray(null);

    this.srcTex = this.texture2D();
    this.srcW = 0;
    this.srcH = 0;
    this.depthTex = this.texture2D();   // the current picture's AI depth (one byte a pixel), when it has one
    this.capTex = this.texture2D();     // the caption over the picture, drawn at the output's size
    this.capKey = "";
    this.capCanvas = null;
    this.maskTex = this.texture2D();    // the followed object in the current picture (one byte a pixel)
    this.maskKey = null;
    this.maskSize = "";
    this.trackPoint = null;             // its centre: [u, v, nearness], or null where it isn't seen
    this.maskVol = null;
    this.pose = identity();             // the current picture's camera (camera-to-world), and its time
    this.poseTime = 0;
    this.poseTex = null;
    this.depthKey = null;
    this.depthSize = "";
    this.writeFbo = gl.createFramebuffer();
    this.vol = null;
    this.head = 0;
    this.targets = new Map();  // "w x h" -> buffers for that output size
    this.outFbo = null;
  }

  texture2D(filter) {
    const gl = this.gl;
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter || gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }

  // ---- the volume ------------------------------------------------------------------------------------------
  // w x h texels per frame, n frames of history. Contents are lost: fill() it afterwards.
  setVolume(w, h, n) {
    const gl = this.gl;
    if (this.vol && this.volW === w && this.volH === h && this.volN === n) return false;
    if (this.vol) gl.deleteTexture(this.vol);
    this.vol = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_3D, this.vol);
    gl.texStorage3D(gl.TEXTURE_3D, 1, gl.RGBA8, w, h, n);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.REPEAT);  // the ring buffer wraps in depth
    // the followed object's mask in every frame, laid out like the volume
    if (this.maskVol) gl.deleteTexture(this.maskVol);
    this.maskVol = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_3D, this.maskVol);
    gl.texStorage3D(gl.TEXTURE_3D, 1, gl.R8, w, h, n);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.REPEAT);
    this.layerTrack = new Float32Array(4 * n);   // per layer: the object's u, v, nearness, and 1 where it was seen
    this.volW = w;
    this.volH = h;
    this.volN = n;
    this.head = 0;
    // every layer's camera: on the GPU (float texture, 4 texels a layer) and here (for the view and the path line)
    if (this.poseTex) gl.deleteTexture(this.poseTex);
    this.poseTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.poseTex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, 4, n);
    for (const k of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, k, gl.NEAREST);
    this.poses = new Float32Array(16 * n);
    this.layerTimes = new Float64Array(n);
    for (let i = 0; i < n; i++) this.poses.set(identity(), i * 16);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 4, n, gl.RGBA, gl.FLOAT, this.poses);
    return true;
  }

  // The camera the current picture was seen from (camera-to-world; null: where the first frame was), and the time of
  // the picture (seconds, so the layer's camera can be worked out again when the path changes).
  setPose(m, time = 0) {
    this.pose = m || identity();
    this.poseTime = time;
  }

  // Every layer gets its camera again from its time: poseOf(seconds) -> matrix | null. For a changed path.
  repose(poseOf) {
    if (!this.vol) return;
    for (let i = 0; i < this.volN; i++) this.poses.set(poseOf(this.layerTimes[i]) || identity(), i * 16);
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.poseTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 4, this.volN, gl.RGBA, gl.FLOAT, this.poses);
  }

  // Upload a source picture: an <img>, <video>, <canvas>, ImageBitmap or { data: Uint8Array, width, height }.
  setSource(src, width, height) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.srcTex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    if (src && src.data) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, src.width, src.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, src.data);
      this.srcW = src.width;
      this.srcH = src.height;
    } else {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, src);
      this.srcW = width;
      this.srcH = height;
    }
    // Mipmaps, so a big video shrinks into the small volume without shimmer.
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.generateMipmap(gl.TEXTURE_2D);
  }

  // The AI depth that goes with the current source picture: { data: Uint8Array (255 = nearest), width, height },
  // or null when it has none. `key` names the depth frame, so the same one isn't uploaded twice.
  setDepth(d, key = null) {
    if (!d) { this.depthKey = null; return; }
    if (key !== null && key === this.depthKey) return;
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.depthTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);   // rows of any width
    const size = `${d.width}x${d.height}`;
    if (size === this.depthSize) {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, d.width, d.height, gl.RED, gl.UNSIGNED_BYTE, d.data);
    } else {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, d.width, d.height, 0, gl.RED, gl.UNSIGNED_BYTE, d.data);
      this.depthSize = size;
    }
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    this.depthKey = key ?? {};   // a fresh object: never equal to the next key
  }

  // The caption for a w x h output: drawn into a canvas (a dark box with light text, low in the middle) and kept
  // as a texture until the text or the size changes.
  caption(text, w, h) {
    const key = `${text}|${w}x${h}`;
    if (key === this.capKey) return;
    this.capKey = key;
    if (!this.capCanvas) this.capCanvas = document.createElement("canvas");
    const c = this.capCanvas;
    c.width = w;
    c.height = h;
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    const size = Math.max(10, Math.round(Math.min(w, h) * 0.034)), pad = Math.round(size * 0.6);
    ctx.font = `500 ${size}px "Segoe UI", system-ui, sans-serif`;
    const maxW = w * 0.84 - pad * 2, lines = [];
    let line = "";
    for (const word of text.split(/\s+/)) {
      const next = line ? `${line} ${word}` : word;
      if (line && ctx.measureText(next).width > maxW) { lines.push(line); line = word; } else line = next;
    }
    if (line) lines.push(line);
    const lh = Math.round(size * 1.32);
    const bw = Math.min(w * 0.88, Math.max(...lines.map((l) => ctx.measureText(l).width)) + pad * 2);
    const bh = lines.length * lh + pad * 1.4;
    const x = (w - bw) / 2, y = h - h * 0.065 - bh, r = size * 0.45;
    ctx.fillStyle = "rgba(10, 10, 12, 0.66)";
    ctx.beginPath();
    ctx.roundRect(x, y, bw, bh, r);
    ctx.fill();
    ctx.fillStyle = "#f4f2ee";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    lines.forEach((l, i) => ctx.fillText(l, w / 2, y + pad * 0.7 + lh * (i + 0.5)));
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.capTex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);              // the compose pass reads bottom-up
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, c);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
  }

  // The followed object in the current picture: { data: Uint8Array (255 = the object), width, height } or null,
  // and its centre [u, v, nearness] (null where it isn't seen).
  setMask(m, key = null, point = null) {
    this.trackPoint = point;
    if (!m) { this.maskKey = null; return; }
    if (key !== null && key === this.maskKey) return;
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.maskTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    const size = `${m.width}x${m.height}`;
    if (size === this.maskSize) {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, m.width, m.height, gl.RED, gl.UNSIGNED_BYTE, m.data);
    } else {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, m.width, m.height, 0, gl.RED, gl.UNSIGNED_BYTE, m.data);
      this.maskSize = size;
    }
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    this.maskKey = key ?? {};
  }

  // Draw the current source picture (and its depth) into one layer of the volume.
  writeLayer(layer) {
    const gl = this.gl, P = this.progs.write;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.writeFbo);
    gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, this.vol, 0, layer);
    gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, this.maskVol, 0, layer);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.viewport(0, 0, this.volW, this.volH);
    gl.disable(gl.BLEND);
    gl.disable(gl.CULL_FACE);
    gl.useProgram(P.p);
    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, this.depthTex);
    gl.uniform1i(P.u.uDepth, 4);
    gl.uniform1i(P.u.uHasDepth, this.depthKey !== null ? 1 : 0);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this.maskTex);
    gl.uniform1i(P.u.uMask, 3);
    gl.uniform1i(P.u.uHasMask, this.maskKey !== null ? 1 : 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.srcTex);
    gl.uniform1i(P.u.uSrc, 0);
    gl.uniform2f(P.u.uSize, this.volW, this.volH);
    // cover fit: crop the source to the volume's shape
    const sa = this.srcW / Math.max(1, this.srcH), va = this.volW / this.volH;
    let cw = 1, ch = 1;
    if (sa > va) cw = va / sa; else ch = sa / va;
    gl.uniform4f(P.u.uCrop, (1 - cw) / 2, (1 - ch) / 2, cw, ch);
    gl.bindVertexArray(this.emptyVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.poses.set(this.pose, layer * 16);
    this.layerTimes[layer] = this.poseTime;
    const tp = this.trackPoint;
    this.layerTrack.set(tp ? [tp[0], tp[1], tp[2], 1] : [0, 0, 0, 0], layer * 4);
    gl.bindTexture(gl.TEXTURE_2D, this.poseTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, layer, 4, 1, gl.RGBA, gl.FLOAT, this.pose instanceof Float32Array ? this.pose : new Float32Array(this.pose));
  }

  // The current source picture becomes the newest frame.
  push() {
    this.head = (this.head + 1) % this.volN;
    this.writeLayer(this.head);
  }

  // Every layer becomes the current source picture (a fresh start with no history yet).
  fill() {
    for (let i = 0; i < this.volN; i++) this.writeLayer(i);
    this.head = 0;
  }

  // ---- output buffers --------------------------------------------------------------------------------------
  target(w, h) {
    const key = `${w}x${h}`;
    let t = this.targets.get(key);
    if (t) return t;
    const gl = this.gl;
    // Keep at most the preview size and one export size around.
    if (this.targets.size >= 2) {
      const [oldKey, old] = this.targets.entries().next().value;
      this.freeTarget(old);
      this.targets.delete(oldKey);
    }
    t = { w, h };
    t.msaa = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, t.msaa);
    gl.renderbufferStorageMultisample(gl.RENDERBUFFER, this.samples, gl.RGBA8, w, h);
    t.msaaFbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.msaaFbo);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, t.msaa);
    const tex = (tw, th) => {
      const x = this.texture2D();
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, tw, th, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      const f = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, f);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, x, 0);
      return { tex: x, fbo: f, w: tw, h: th };
    };
    t.scene = tex(w, h);
    const gw = Math.max(1, Math.round(w / 4)), gh = Math.max(1, Math.round(h / 4));
    t.glowA = tex(gw, gh);
    t.glowB = tex(gw, gh);
    t.out = tex(w, h);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.targets.set(key, t);
    return t;
  }

  freeTarget(t) {
    const gl = this.gl;
    gl.deleteRenderbuffer(t.msaa);
    gl.deleteFramebuffer(t.msaaFbo);
    for (const k of ["scene", "glowA", "glowB", "out"]) {
      gl.deleteTexture(t[k].tex);
      gl.deleteFramebuffer(t[k].fbo);
    }
  }

  // ---- drawing ---------------------------------------------------------------------------------------------
  // A grid of `n` cells along the front face's long side, square cells: [across, down].
  cells(n) {
    const a = this.volW / this.volH;
    return a >= 1 ? [n, Math.max(1, Math.round(n / a))] : [Math.max(1, Math.round(n * a)), n];
  }

  // The cube's half-size: the front face keeps the source's shape, its long side is 1; depth is relative to that.
  half(depth) {
    const a = this.volW / this.volH;
    const m = Math.max(a, 1);
    return [0.5 * a / m, 0.5 / m, 0.5 * depth];
  }

  matrices(s, w, h) {
    const half = this.half(s.depth);
    // your mouse turn is applied on screen axes, on top of the choreography
    const rot = multiply(rotation(s.dragYaw || 0, s.dragPitch || 0, 0), rotation(s.yaw, s.pitch, s.roll));
    // the block turns around its middle, or around the newest frame (which then stays in the middle of the picture)
    const dNow = s.newestBack ? 1 : 0;
    const centre = s.pivot === "newest" ? shapePoint(s, half, [0, 0, half[2] * (1 - 2 * dNow)], dNow)
      : shapePoint(s, half, [0, 0, 0], 0.5);
    const model = multiply(rot, translate(-centre[0], -centre[1], -centre[2]));
    const aspect = w / h;
    const fovY = (s.fov * Math.PI) / 180;
    const fovX = 2 * Math.atan(Math.tan(fovY / 2) * aspect);
    const radius = Math.hypot(half[0], half[1] + (s.shape === "stack" ? riseOf(half) / 2 : 0), half[2]);
    const dist = radius / Math.sin(Math.min(fovX, fovY) / 2) / Math.max(0.05, s.zoom);
    const proj = perspective(fovY, aspect, Math.max(0.01, dist - radius * 4), dist + radius * 4);
    const view = translate(s.panX * radius, s.panY * radius, -dist);
    const mvp = multiply(proj, multiply(view, model));
    const camObj = unrotate(rot, [-s.panX * radius, -s.panY * radius, dist]);  // the camera, seen from the cube
    camObj[2] += centre[2];
    return { half, model, mvp, camObj };
  }

  bindVol(P, s) {
    const gl = this.gl, t = s.tone;
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_3D, this.vol);
    gl.uniform1i(P.u.uVol, 1);
    gl.uniform1f(P.u.uHead, this.head);
    gl.uniform1f(P.u.uLayers, this.volN);
    gl.uniform1f(P.u.uFlip, s.newestBack ? 1 : 0);
    gl.uniform2f(P.u.uVolTexel, 1 / this.volW, 1 / this.volH);
    gl.uniform1f(P.u.uMirror, t.mirror ? 1 : 0);
    gl.uniform1f(P.u.uSoften, t.soften);
    gl.uniform1f(P.u.uToneB, t.brightness);
    gl.uniform1f(P.u.uToneC, t.contrast);
    gl.uniform1f(P.u.uToneS, t.saturation);
    const r = s.relief || {};
    gl.uniform1f(P.u.uAi, r.ai ? 1 : 0);
    gl.uniform1f(P.u.uRelief, r.amount || 0);
    gl.uniform1f(P.u.uTrimFar, r.trimFar || 0);
    gl.uniform1f(P.u.uTrimDark, r.trimDark || 0);
    gl.uniform1f(P.u.uMotion, r.motion || 0);
    gl.uniform1f(P.u.uFadeOld, s.fadeOld || 0);
    gl.activeTexture(gl.TEXTURE6);
    gl.bindTexture(gl.TEXTURE_3D, this.maskVol);
    gl.uniform1i(P.u.uMaskVol, 6);
    gl.uniform1f(P.u.uTrackDim, s.track?.dim || 0);
    gl.uniform3fv(P.u.uDimTo, s.background || [0, 0, 0]);
  }

  // The camera's tan(half its field of view) across and down, from the lens across the long side.
  tanHalf(fovDeg) {
    const t = Math.tan((fovDeg * Math.PI) / 360), a = this.volW / this.volH;
    return a >= 1 ? [t, t / a] : [t * a, t];
  }

  // The Camera path view: the world seen from the newest frame's camera, turned around a pivot in front of it (or
  // around the middle of the frames), the source picture fitted into the output.
  pathMatrices(s, w, h) {
    const zp = s.path.zp;
    const now = this.poses.subarray(this.head * 16, this.head * 16 + 16);
    const toCam = invertRigid(now);
    let pivot = [0, 0, -zp];
    if (s.pivot !== "newest") {
      const c = [0, 0, 0];
      for (let i = 0; i < this.volN; i++) {
        const q = transformPoint(this.poses.subarray(i * 16, i * 16 + 16), [0, 0, -zp]);
        for (let k = 0; k < 3; k++) c[k] += q[k] / this.volN;
      }
      pivot = transformPoint(toCam, c);
    }
    const dist = Math.max(zp * 0.25, Math.hypot(...pivot));
    const rot = multiply(rotation(s.dragYaw || 0, s.dragPitch || 0, 0), rotation(s.yaw, s.pitch, s.roll));
    const model = multiply(rot, multiply(translate(-pivot[0], -pivot[1], -pivot[2]), toCam));
    const aspect = w / h;
    const [tx, ty] = this.tanHalf(s.path.fov);
    const tanY = Math.max(ty, tx / aspect) / Math.max(0.05, s.zoom);
    const proj = perspective(2 * Math.atan(tanY), aspect, 0.01, 5000);
    const view = translate(s.panX * dist, s.panY * dist, -dist);
    return { model, mvp: multiply(proj, multiply(view, model)), focal: h / 2 / tanY, tan: [tx, ty] };
  }

  bindPath(P, s, M) {
    const gl = this.gl, on = s.layout === "path";
    gl.uniform1i(P.u.uLayout, on ? 1 : 0);
    gl.activeTexture(gl.TEXTURE5);
    gl.bindTexture(gl.TEXTURE_2D, this.poseTex);
    gl.uniform1i(P.u.uPoses, 5);
    if (!on) return;
    gl.uniform2fv(P.u.uTanHalf, M.tan);
    gl.uniform1f(P.u.uDepthC, DEPTH_C);
    gl.uniform1f(P.u.uPivotZ, s.path.zp);
  }

  // The path line through the cameras of the frames in the cube (oldest to newest) and the outline of now.
  drawPath(s, mvp, w, h, px) {
    const gl = this.gl, P = this.progs.line, N = this.volN, zp = s.path.zp;
    const segs = [];
    let prev = null;
    for (let k = N - 1; k >= 0; k--) {
      const L = (((this.head - k) % N) + N) % N;
      const c = [this.poses[L * 16 + 12], this.poses[L * 16 + 13], this.poses[L * 16 + 14]];
      if (prev) segs.push(...prev, ...c);
      prev = c;
    }
    const nPath = segs.length / 6;
    const now = this.poses.subarray(this.head * 16, this.head * 16 + 16);
    const [tx, ty] = this.tanHalf(s.path.fov);
    const corner = (x, y) => transformPoint(now, [x * tx * zp, y * ty * zp, -zp]);
    const q = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)];
    for (let i = 0; i < 4; i++) segs.push(...q[i], ...q[(i + 1) % 4]);
    gl.useProgram(P.p);
    gl.bindVertexArray(this.lineVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(segs), gl.DYNAMIC_DRAW);
    gl.uniformMatrix4fv(P.u.uMVP, false, mvp);
    gl.uniform2f(P.u.uViewport, w, h);
    gl.uniform1f(P.u.uWidth, 2.2 * px);
    if (nPath) {
      gl.uniform4f(P.u.uColor, 0.94, 0.63, 0.31, 0.75);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, nPath);
    }
    // the outline of now, drawn by its own instances
    gl.uniform1f(P.u.uWidth, 2.6 * px);
    gl.uniform4f(P.u.uColor, 1.0, 0.6, 0.24, 0.95);
    gl.vertexAttribPointer(gl.getAttribLocation(P.p, "aA"), 3, gl.FLOAT, false, 24, nPath * 24);
    gl.vertexAttribPointer(gl.getAttribLocation(P.p, "aB"), 3, gl.FLOAT, false, 24, nPath * 24 + 12);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, 4);
    gl.vertexAttribPointer(gl.getAttribLocation(P.p, "aA"), 3, gl.FLOAT, false, 24, 0);
    gl.vertexAttribPointer(gl.getAttribLocation(P.p, "aB"), 3, gl.FLOAT, false, 24, 12);
    gl.bindVertexArray(this.emptyVao);
  }

  // The followed object's centre in every frame of the cube, oldest to newest, as line segments (world positions);
  // a frame where it wasn't seen breaks the line.
  trackSegments(s, M) {
    const N = this.volN, half = this.half(s.depth), path = s.layout === "path";
    const [tx, ty] = path ? M.tan : [0, 0];
    const ai = !!s.relief?.ai, relief = s.relief?.amount || 0;
    const segs = [];
    let prev = null;
    for (let k = N - 1; k >= 0; k--) {
      const L = (((this.head - k) % N) + N) % N;
      if (!this.layerTrack[L * 4 + 3]) { prev = null; continue; }
      let u = this.layerTrack[L * 4];
      const v = this.layerTrack[L * 4 + 1], near = ai ? this.layerTrack[L * 4 + 2] : 0.5;
      if (s.tone?.mirror) u = 1 - u;
      let p;
      if (path) {
        const z = s.path.zp + (distance(near) - s.path.zp) * relief;
        p = transformPoint(this.poses.subarray(L * 16, L * 16 + 16), [(u * 2 - 1) * tx * z, (1 - v * 2) * ty * z, -z]);
      } else {
        const d = s.newestBack ? 1 - k / Math.max(N - 1, 1) : k / Math.max(N - 1, 1);
        p = shapePoint(s, half, [(u * 2 - 1) * half[0], (1 - v * 2) * half[1],
          half[2] * (1 - 2 * d) + relief * (near - 0.5) * 0.5], d);
      }
      if (prev) segs.push(...prev, ...p);
      prev = p;
    }
    return segs;
  }

  drawTrack(s, M, w, h, px) {
    const segs = this.trackSegments(s, M);
    if (!segs.length) return;
    const gl = this.gl, P = this.progs.line;
    gl.useProgram(P.p);
    gl.bindVertexArray(this.lineVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(segs), gl.DYNAMIC_DRAW);
    gl.uniformMatrix4fv(P.u.uMVP, false, M.mvp);
    gl.uniform2f(P.u.uViewport, w, h);
    gl.uniform1f(P.u.uWidth, 3.4 * px);
    const c = s.track.color;
    gl.uniform4f(P.u.uColor, c[0], c[1], c[2], 1);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, segs.length / 6);
    gl.bindVertexArray(this.emptyVao);
  }

  bindShape(P, s, half) {
    const gl = this.gl;
    gl.uniform1i(P.u.uShape, SHAPES.indexOf(s.shape) > 0 ? SHAPES.indexOf(s.shape) : 0);
    gl.uniform1f(P.u.uSpiral, ((s.spiral || 0) * Math.PI) / 180);
    gl.uniform1f(P.u.uRise, riseOf(half));
  }

  // Draw a frame. s: the full look (see main.js). w, h: output size in pixels.
  // toCanvas: draw into the window's canvas; otherwise the result stays in this.outFbo for readPixels.
  render(s, w, h, toCanvas = true) {
    const gl = this.gl;
    if (!this.vol) return;
    const t = this.target(w, h);
    const path = s.layout === "path";
    const M = path ? this.pathMatrices(s, w, h) : this.matrices(s, w, h);
    const { model, mvp } = M;
    const half = this.half(s.depth);
    const camObj = path ? [0, 0, 1] : M.camObj;      // along the path: always oldest first
    const wts = { ...lookWeights(s.morph) };
    if (path) { wts.slices += wts.solid; wts.solid = 0; }   // the block has no sides along a path: sheets instead
    const px = Math.min(w, h) / 1080;  // sizes in the UI are pixels of a 1080-wide reel

    gl.bindFramebuffer(gl.FRAMEBUFFER, t.msaaFbo);
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);  // everything is premultiplied
    gl.disable(gl.CULL_FACE);
    gl.bindVertexArray(this.emptyVao);

    const lifted = Math.abs(s.relief?.amount || 0) > 0.001;
    if (wts.slices > 0.001) {
      const P = this.progs.slice;
      gl.useProgram(P.p);
      this.bindVol(P, s);
      gl.uniformMatrix4fv(P.u.uMVP, false, mvp);
      gl.uniform3fv(P.u.uHalf, half);
      const planes = Math.max(2, Math.min(MAX_PLANES, Math.round(s.slices.planes)));
      gl.uniform1i(P.u.uPlanes, planes);
      gl.uniform1i(P.u.uReverse, camObj[2] > 0 ? 1 : 0);
      gl.uniform1f(P.u.uAlpha, s.slices.opacity * wts.slices);
      gl.uniform1f(P.u.uClear, s.slices.clear);
      // flat slices are one quad; with relief, a grid fine enough to follow the depth
      const [gx, gy] = lifted || path ? this.cells(SLICE_CELLS) : [1, 1];
      gl.uniform2i(P.u.uGrid, gx, gy);
      this.bindShape(P, s, half);
      this.bindPath(P, s, M);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6 * gx * gy, planes);
    }
    if (wts.points > 0.001) {
      const P = this.progs.points;
      gl.useProgram(P.p);
      this.bindVol(P, s);
      gl.uniformMatrix4fv(P.u.uMVP, false, mvp);
      gl.uniform3fv(P.u.uHalf, half);
      const planes = Math.max(2, Math.min(MAX_PLANES, Math.round(s.points.planes)));
      const [gx, gy] = this.cells(Math.max(8, Math.min(MAX_POINT_GRID, Math.round(s.points.density))));
      gl.uniform1i(P.u.uPlanes, planes);
      gl.uniform1i(P.u.uReverse, camObj[2] > 0 ? 1 : 0);
      gl.uniform2i(P.u.uGrid, gx, gy);
      gl.uniform1f(P.u.uSize, s.points.size);
      gl.uniform1f(P.u.uFocal, path ? M.focal : h / 2 / Math.tan((s.fov * Math.PI) / 360));   // pixels per unit at 1
      gl.uniform1f(P.u.uAlpha, s.points.opacity * wts.points);
      this.bindShape(P, s, half);
      this.bindPath(P, s, M);
      gl.drawArrays(gl.POINTS, 0, planes * gx * gy);
    }
    if (wts.wire > 0.001) {
      const P = this.progs.wire;
      gl.useProgram(P.p);
      this.bindVol(P, s);
      gl.uniformMatrix4fv(P.u.uMVP, false, mvp);
      gl.uniform3fv(P.u.uHalf, half);
      const g = wireGrid(path ? { ...s.wire, axis: "frames" } : s.wire, half);   // along a path: a grid per frame
      const segs = g.ga * (g.gb + 1) + (g.ga + 1) * g.gb + (s.wire.diagonals ? g.ga * g.gb : 0);
      const planes = Math.max(2, Math.min(MAX_PLANES, Math.round(s.wire.planes)));
      gl.uniform2i(P.u.uGrid, g.ga, g.gb);
      gl.uniformMatrix3fv(P.u.uAxes, false, g.axes);
      gl.uniform3fv(P.u.uNormal, g.normal);
      gl.uniform1i(P.u.uSegs, segs);
      gl.uniform1i(P.u.uPlanes, planes);
      gl.uniform1f(P.u.uWidth, s.wire.width * px);
      gl.uniform1f(P.u.uDisp, s.wire.wave);
      gl.uniform2f(P.u.uViewport, w, h);
      gl.uniform1f(P.u.uAlpha, s.wire.opacity * wts.wire);
      gl.uniform1f(P.u.uInkAmt, s.wire.ink);
      gl.uniform3fv(P.u.uInk, s.wire.inkColor);
      gl.uniform1f(P.u.uContrast, s.wire.contrast);
      this.bindShape(P, s, half);
      this.bindPath(P, s, M);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, segs * planes);
    }
    if (wts.solid > 0.001) {
      const P = this.progs.solid;
      gl.useProgram(P.p);
      this.bindVol(P, s);
      gl.uniformMatrix4fv(P.u.uMVP, false, mvp);
      gl.uniformMatrix4fv(P.u.uModel, false, model);
      gl.uniform3fv(P.u.uHalf, half);
      gl.uniform1f(P.u.uAlpha, wts.solid);
      gl.uniform1f(P.u.uShade, s.solid.shade);
      this.bindShape(P, s, half);
      gl.enable(gl.CULL_FACE);
      gl.cullFace(gl.BACK);
      gl.bindVertexArray(this.cubeVao);
      gl.drawArrays(gl.TRIANGLES, 0, this.cubeCount);
      gl.disable(gl.CULL_FACE);
      gl.bindVertexArray(this.emptyVao);
    }

    if (path && s.path.showPath) this.drawPath(s, mvp, w, h, px);
    if (s.track?.line) this.drawTrack(s, M, w, h, px);

    // resolve the multisampled scene
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, t.msaaFbo);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, t.scene.fbo);
    gl.blitFramebuffer(0, 0, w, h, 0, 0, w, h, gl.COLOR_BUFFER_BIT, gl.NEAREST);
    gl.disable(gl.BLEND);

    // glow: bright parts at quarter size, blurred twice
    const glowOn = s.glow > 0.001;
    if (glowOn) {
      let P = this.progs.bright;
      gl.useProgram(P.p);
      gl.bindFramebuffer(gl.FRAMEBUFFER, t.glowA.fbo);
      gl.viewport(0, 0, t.glowA.w, t.glowA.h);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, t.scene.tex);
      gl.uniform1i(P.u.uTex, 0);
      gl.uniform2f(P.u.uTexel, 1 / w, 1 / h);
      gl.uniform1f(P.u.uThr, s.glowThreshold);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      P = this.progs.blur;
      gl.useProgram(P.p);
      gl.uniform1i(P.u.uTex, 0);
      const r = Math.max(0.2, s.glowSize) * (Math.min(t.glowA.w, t.glowA.h) / 270);
      for (const scale of [1, 2.2]) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, t.glowB.fbo);
        gl.bindTexture(gl.TEXTURE_2D, t.glowA.tex);
        gl.uniform2f(P.u.uStep, (r * scale) / t.glowA.w, 0);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        gl.bindFramebuffer(gl.FRAMEBUFFER, t.glowA.fbo);
        gl.bindTexture(gl.TEXTURE_2D, t.glowB.tex);
        gl.uniform2f(P.u.uStep, 0, (r * scale) / t.glowA.h);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      }
    }

    // compose over the background
    const P = this.progs.compose;
    gl.useProgram(P.p);
    gl.bindFramebuffer(gl.FRAMEBUFFER, toCanvas ? null : t.out.fbo);
    gl.viewport(0, 0, w, h);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, t.scene.tex);
    gl.uniform1i(P.u.uScene, 0);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, t.glowA.tex);
    gl.uniform1i(P.u.uGlow, 2);
    gl.uniform3fv(P.u.uBg, s.background);
    gl.uniform1i(P.u.uOut, { checker: 1, alpha: 2 }[s.output] || 0);
    gl.uniform1f(P.u.uCheck, Math.max(4, Math.round(18 * px)));
    gl.uniform1f(P.u.uGlowAmt, glowOn ? s.glow * 1.6 : 0);
    gl.uniform1f(P.u.uGrain, s.grain);
    gl.uniform1f(P.u.uSeed, (s.seed || 0) % 1000);
    if (s.caption) this.caption(s.caption, w, h);
    gl.activeTexture(gl.TEXTURE7);
    gl.bindTexture(gl.TEXTURE_2D, this.capTex);
    gl.uniform1i(P.u.uCap, 7);
    gl.uniform1i(P.u.uHasCap, s.caption ? 1 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.outFbo = toCanvas ? null : t.out.fbo;
    gl.bindVertexArray(null);
  }

  // RGBA pixels of the last off-screen render (bottom row first, as WebGL reads them).
  read(w, h, out) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.outFbo);
    out = out || new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, out);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return out;
  }
}
