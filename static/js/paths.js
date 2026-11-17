// Camera paths: where the camera was for each moment, as camera-to-world matrices (column-major, OpenGL axes: x right,
// y up, the camera looking down -z). "video" follows the path measured from the video (path.py, with its AI depth);
// Orbit, Walk and Turn are made up, for clips whose camera held still.
import { identity, multiply, rotY, translate } from "./mat.js";

export const PATH_KINDS = ["video", "orbit", "walk", "turn"];
export const DEPTH_C = 0.08;   // nearness 0..1 -> distance, the rule path.py measures with

// How far away a point of nearness n (0 far .. 1 near) is, in the path's unit.
export const distance = (near) => 1 / (DEPTH_C + (1 - DEPTH_C) * near);

// A made-up path at time t (seconds, already times the path speed). zp: how far away the middle of the scene is.
export function madePose(kind, t, zp) {
  if (kind === "walk") return translate(0, 0, -0.3 * zp * t);            // forward, a third of the scene a second
  if (kind === "turn") return rotY((-14 * t * Math.PI) / 180);           // turning right on the spot
  if (kind === "orbit") {                                                 // round the middle of the scene
    const a = (-20 * t * Math.PI) / 180;
    return multiply(translate(zp * Math.sin(a), 0, zp * (Math.cos(a) - 1)), rotY(a));
  }
  return identity();
}

// Gram-Schmidt on the rotation part of a camera-to-world matrix (after blending two of them).
function orthonormalize(m) {
  const n = (x, y, z) => { const l = Math.hypot(x, y, z) || 1; return [x / l, y / l, z / l]; };
  const x = n(m[0], m[1], m[2]);
  const d = x[0] * m[4] + x[1] * m[5] + x[2] * m[6];
  const y = n(m[4] - d * x[0], m[5] - d * x[1], m[6] - d * x[2]);
  const z = [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]];
  m.set([...x, 0, ...y, 0, ...z, 0], 0);
  m[15] = 1;
  return m;
}

// The measured path ({ fps, frames, poses: Float32Array(frames * 16) }) at a moment of the video (seconds).
export function measuredPose(path, t) {
  const x = Math.max(0, Math.min(path.frames - 1, t * path.fps));
  const i = Math.floor(x), j = Math.min(path.frames - 1, i + 1), f = x - i;
  const m = new Float32Array(16);
  for (let k = 0; k < 16; k++) m[k] = path.poses[i * 16 + k] * (1 - f) + path.poses[j * 16 + k] * f;
  return f > 0 ? orthonormalize(m) : m;
}

// The top-down map of a path: track = [{ x, z, hx, hz }] (position and heading, x across, forward up the map),
// span = [first, last] index of the stretch the cube holds now (the last one is now).
export function drawMap(canvas, track, span) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height, css = getComputedStyle(canvas);
  const col = (name, fb) => css.getPropertyValue(name).trim() || fb;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = col("--field", "#222226");
  ctx.fillRect(0, 0, W, H);
  if (!track.length) return;
  let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity;
  for (const p of track) { x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); z0 = Math.min(z0, p.z); z1 = Math.max(z1, p.z); }
  const pad = 14;   // the path as big as the map allows, the same scale both ways
  const k = Math.min((W - 2 * pad) / Math.max(x1 - x0, 1e-3), (H - 2 * pad) / Math.max(z1 - z0, 1e-3));
  const ox = W / 2 - ((x0 + x1) / 2) * k, oz = H / 2 + ((z0 + z1) / 2) * k;
  const at = (p) => [ox + p.x * k, oz - p.z * k];
  // grid
  ctx.strokeStyle = col("--line", "#2d2d32");
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let g = 0; g <= W; g += 20) { ctx.moveTo(g + 0.5, 0); ctx.lineTo(g + 0.5, H); }
  for (let g = 0; g <= H; g += 20) { ctx.moveTo(0, g + 0.5); ctx.lineTo(W, g + 0.5); }
  ctx.stroke();
  const line = (from, to, style, width) => {
    ctx.strokeStyle = style;
    ctx.lineWidth = width;
    ctx.beginPath();
    for (let i = from; i <= to; i++) { const [x, y] = at(track[i]); i === from ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
    ctx.stroke();
  };
  line(0, track.length - 1, col("--muted", "#95938d"), 1.2);
  const [a, b] = [Math.max(0, span[0]), Math.min(track.length - 1, span[1])];
  if (b > a) line(a, b, "#f0a050", 2.4);
  const now = track[b];
  const [nx, ny] = at(now);
  ctx.strokeStyle = "#f0a050";
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.moveTo(nx, ny);
  ctx.lineTo(nx + now.hx * 14, ny - now.hz * 14);
  ctx.stroke();
  ctx.fillStyle = "#f0a050";
  ctx.beginPath();
  ctx.arc(nx, ny, 3.5, 0, Math.PI * 2);
  ctx.fill();
}
