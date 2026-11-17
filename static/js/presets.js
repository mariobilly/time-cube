// Every setting of the cube, in the order the side panel shows them, plus the built-in presets.
// type: range (default) | bool | choice | color | depth (the AI depth button and its progress) | pathmap (the
// camera path's map) | track (finding and following an object) | aisees (captions and questions). The panel is generated from this list; show(S) hides a control (a section: its third item).
const inPath = (S) => S.layout === "path";
const inCube = (S) => S.layout !== "path";

export const SECTIONS = [
  ["look", "Look"],
  ["path", "Camera path", inPath],
  ["depth", "Depth"],
  ["track", "Track"],
  ["ai", "What the AI sees"],
  ["moves", "Moves"],
  ["time", "Time"],
  ["wire", "Wire"],
  ["slices", "Slices"],
  ["solid", "Solid", inCube],
  ["points", "Points"],
  ["finish", "Finish"],
  ["source", "Source"],
];

export const PARAMS = [
  { id: "layout", sec: "look", type: "choice", label: "Layout", def: "cube",
    options: [["cube", "Time cube"], ["path", "Camera path"]],
    hint: "Time cube stacks the frames in a block; Camera path puts each one where the camera was" },
  { id: "morph", sec: "look", label: "Wire · Slices · Solid · Points", min: 0, max: 3, step: 0.01, def: 1.7,
    hint: "0 is all wire, 1 all slices, 2 the solid block, 3 clouds of points" },
  { id: "autoMorph", sec: "look", type: "bool", label: "Morph with the moves", def: true,
    hint: "Tour and Tumble change the look as the cube turns" },

  { id: "pathMap", sec: "path", type: "pathmap", label: "Path" },
  { id: "pathFrom", sec: "path", type: "choice", label: "Path from", def: "video",
    options: [["video", "From video"], ["orbit", "Orbit"], ["walk", "Walk"], ["turn", "Turn"]],
    hint: "From video is measured from the video (it needs its AI depth). Orbit, Walk and Turn are made up, for a camera that held still" },
  { id: "pathSpeed", sec: "path", label: "Speed along the path", min: 0.1, max: 3, step: 0.05, def: 1, unit: "×",
    show: (S) => S.pathFrom !== "video" },
  { id: "camFov", sec: "path", label: "Camera lens", min: 30, max: 110, step: 1, def: 65, unit: "°",
    hint: "The video's field of view across its long side (phones: about 65–75°). The path is measured again for it" },
  { id: "showPath", sec: "path", type: "bool", label: "Show the path", def: true,
    hint: "The line the camera took, and an outline around now" },

  { id: "depthBuild", sec: "depth", type: "depth", label: "AI depth" },
  { id: "depthSrc", sec: "depth", type: "choice", label: "Depth from", def: "bright",
    options: [["ai", "AI depth"], ["bright", "Brightness"]],
    hint: "Brightness works for every source: dark parts count as near" },
  { id: "relief", sec: "depth", label: "Relief", min: 0, max: 2, step: 0.01, def: 0,
    hint: "Near things come forward, far things go back. 0 keeps every frame flat (along a camera path, 1 is true depth)" },
  { id: "trimFar", sec: "depth", label: "Trim far away (sky)", min: 0, max: 0.9, step: 0.01, def: 0 },
  { id: "trimDark", sec: "depth", label: "Remove dark pixels", min: 0, max: 0.9, step: 0.01, def: 0 },
  { id: "motion", sec: "depth", label: "Highlight motion", min: 0, max: 1, step: 0.01, def: 0,
    hint: "What stays still fades away; what moves stays" },

  { id: "trackBox", sec: "track", type: "track", label: "Track" },
  { id: "isolate", sec: "track", type: "bool", label: "Isolate tracked", def: false,
    hint: "Only the followed object stays" },
  { id: "dimRest", sec: "track", label: "Dim the rest", min: 0, max: 1, step: 0.01, def: 0,
    hint: "The rest of the picture fades around the object", show: (S) => !S.isolate },
  { id: "trackLine", sec: "track", type: "bool", label: "Track line", def: true,
    hint: "A glowing line through the object's centre in every frame of the cube" },
  { id: "trackColor", sec: "track", type: "color", label: "Line colour", def: "#ffb04a" },

  { id: "aiBox", sec: "ai", type: "aisees", label: "What the AI sees" },
  { id: "captionOn", sec: "ai", type: "bool", label: "Caption on the picture", def: false,
    hint: "The caption of the moment shown, at the bottom; it goes into PNGs, recordings and exports too" },

  { id: "mode", sec: "moves", type: "choice", label: "Moves", def: "reel",
    options: [["reel", "Tour"], ["tumble", "Tumble"], ["sway", "Sway"], ["spin", "Spin"], ["still", "Still"]],
    hint: "Tour plays a 30 s choreography as a loop; Sway swings around the front, to show the relief. Drag the cube to turn it, scroll to zoom" },
  { id: "pivot", sec: "moves", type: "choice", label: "Turn around", def: "centre",
    options: [["centre", "Cube centre"], ["newest", "Newest frame"]],
    hint: "Newest frame keeps now in the middle while the past swings behind it" },
  { id: "speed", sec: "moves", label: "Speed", min: 0.1, max: 3, step: 0.05, def: 1, unit: "×" },
  { id: "spinYaw", sec: "moves", label: "Spin left / right", min: -180, max: 180, step: 1, def: 24, unit: "°/s", only: "spin" },
  { id: "spinPitch", sec: "moves", label: "Spin up / down", min: -180, max: 180, step: 1, def: 0, unit: "°/s", only: "spin" },
  { id: "spinRoll", sec: "moves", label: "Roll", min: -180, max: 180, step: 1, def: 0, unit: "°/s", only: "spin" },
  { id: "zoom", sec: "moves", label: "Zoom", min: 0.4, max: 3, step: 0.01, def: 1.15 },
  { id: "fov", sec: "moves", label: "Lens", min: 8, max: 90, step: 1, def: 34, unit: "°", hint: "Narrow is flat, wide is dramatic" },

  { id: "history", sec: "time", label: "Frames in the cube", min: 8, max: 180, step: 1, def: 60 },
  { id: "step", sec: "time", label: "Frame step", min: 1, max: 8, step: 1, def: 1,
    hint: "Keep every Nth frame: a longer stretch of time in the same cube" },
  { id: "shape", sec: "time", type: "choice", label: "Shape of time", def: "cube",
    options: [["cube", "Cube"], ["stack", "Stack"], ["echo", "Echo"]],
    hint: "Stack leans the past up behind now; Echo shrinks it toward the middle, like a tunnel", show: (S) => S.layout !== "path" },
  { id: "depth", sec: "time", label: "Cube depth", min: 0.1, max: 3, step: 0.01, def: 1,
    hint: "How long the block of time is, front to back", show: (S) => S.layout !== "path" },
  { id: "spiral", sec: "time", label: "Spiral", min: -360, max: 360, step: 1, def: 0, unit: "°",
    hint: "Each older frame turns a little more around the time axis; this is the turn of the oldest", show: (S) => S.layout !== "path" },
  { id: "newestBack", sec: "time", type: "bool", label: "Newest frame at the back", def: false, show: (S) => S.layout !== "path" },
  { id: "fadeOld", sec: "time", label: "Fade older frames", min: 0, max: 1, step: 0.01, def: 0,
    hint: "The newest frame stays whole; older ones thin out (Wire, Slices, Points)" },
  { id: "quality", sec: "time", type: "choice", label: "Detail", def: 384,
    options: [[256, "Draft"], [384, "Good"], [512, "Best"]] },

  { id: "wireAxis", sec: "wire", type: "choice", label: "Planes through", def: "columns",
    options: [["columns", "Columns"], ["frames", "Frames"], ["rows", "Rows"]],
    hint: "Columns and Rows cut through time (long time-lines); Frames is one plane per moment", show: (S) => S.layout !== "path" },
  { id: "wireGrid", sec: "wire", label: "Grid", min: 4, max: 128, step: 1, def: 22 },
  { id: "wireStretch", sec: "wire", label: "Cell length through time", min: 1, max: 8, step: 0.1, def: 3, unit: "×",
    hint: "Longer cells along the depth make long time-lines", show: (S) => S.layout !== "path" },
  { id: "wirePlanes", sec: "wire", label: "Planes", min: 2, max: 128, step: 1, def: 22 },
  { id: "wireWidth", sec: "wire", label: "Line width", min: 0.3, max: 6, step: 0.1, def: 1.6, unit: "px" },
  { id: "wireOpacity", sec: "wire", label: "Opacity", min: 0, max: 1, step: 0.01, def: 0.7 },
  { id: "wireContrast", sec: "wire", label: "Line contrast", min: 0.5, max: 2.5, step: 0.01, def: 1.3 },
  { id: "wireInk", sec: "wire", label: "Ink", min: 0, max: 1, step: 0.01, def: 0.3,
    hint: "Pulls the line colour toward the ink colour" },
  { id: "wireInkColor", sec: "wire", type: "color", label: "Ink colour", def: "#16130f" },
  { id: "wireDiagonals", sec: "wire", type: "bool", label: "Triangles (diagonals)", def: true },
  { id: "wireWave", sec: "wire", label: "Depth relief", min: -2, max: 2, step: 0.01, def: 0,
    hint: "Dark parts of each plane bulge forward" },

  { id: "slicePlanes", sec: "slices", label: "Planes", min: 2, max: 128, step: 1, def: 48 },
  { id: "sliceOpacity", sec: "slices", label: "Opacity", min: 0, max: 1, step: 0.01, def: 0.14 },
  { id: "sliceClear", sec: "slices", label: "Clear the brights", min: 0, max: 1, step: 0.01, def: 0.45 },

  { id: "solidShade", sec: "solid", label: "Side shading", min: 0, max: 0.6, step: 0.01, def: 0.06 },

  { id: "pointPlanes", sec: "points", label: "Frames shown", min: 2, max: 128, step: 1, def: 40 },
  { id: "pointDensity", sec: "points", label: "Point density", min: 32, max: 320, step: 1, def: 160,
    hint: "Points across the picture's long side" },
  { id: "pointSize", sec: "points", label: "Point size", min: 0.3, max: 3, step: 0.01, def: 1, unit: "×",
    hint: "1 is touching its neighbours" },
  { id: "pointOpacity", sec: "points", label: "Opacity", min: 0, max: 1, step: 0.01, def: 0.8 },

  { id: "background", sec: "finish", type: "color", label: "Background", def: "#eeeeee" },
  { id: "transparent", sec: "finish", type: "bool", label: "Transparent background", def: false,
    hint: "Exports and PNGs keep only the cube and its glow (ProRes 4444 or VP9 alpha); the preview shows a checkerboard" },
  { id: "glow", sec: "finish", label: "Glow", min: 0, max: 1.5, step: 0.01, def: 0.55 },
  { id: "glowSize", sec: "finish", label: "Glow size", min: 0.2, max: 4, step: 0.01, def: 1.4 },
  { id: "glowThreshold", sec: "finish", label: "Glow from", min: 0.2, max: 1, step: 0.01, def: 0.7,
    hint: "Only parts brighter than this glow" },
  { id: "grain", sec: "finish", label: "Grain", min: 0, max: 1, step: 0.01, def: 0.08 },

  { id: "brightness", sec: "source", label: "Brightness", min: -0.4, max: 0.4, step: 0.01, def: 0 },
  { id: "contrast", sec: "source", label: "Contrast", min: 0.4, max: 2, step: 0.01, def: 1 },
  { id: "saturation", sec: "source", label: "Saturation", min: 0, max: 2, step: 0.01, def: 1 },
  { id: "soften", sec: "source", label: "Soften", min: 0, max: 3, step: 0.01, def: 0.5 },
  { id: "mirror", sec: "source", type: "bool", label: "Mirror", def: false },
  { id: "photoHold", sec: "source", label: "Photo hold", min: 1, max: 90, step: 1, def: 8, unit: "frames",
    hint: "How long each photo stays (photos only)" },
];

export function defaults() {
  const o = {};
  for (const p of PARAMS) if (!["depth", "pathmap", "track", "aisees"].includes(p.type)) o[p.id] = p.def;
  return o;
}

export const PRESETS = [
  { id: "classic", name: "Classic", set: {} },
  { id: "ink", name: "Ink sketch", set: {
    mode: "spin", spinYaw: 18, spinPitch: 6, autoMorph: false, morph: 0, wireAxis: "frames", wireInk: 0.85, wireOpacity: 0.62,
    wireGrid: 38, wirePlanes: 26, wireWidth: 1.1, wireWave: 0.25, wireStretch: 1, background: "#f4f3ef", glow: 0.15, grain: 0.05 } },
  { id: "glass", name: "Glass stack", set: {
    mode: "tumble", autoMorph: false, morph: 1, slicePlanes: 72, sliceOpacity: 0.2, sliceClear: 0.65,
    background: "#efefef", glow: 0.7, depth: 1.2 } },
  { id: "slit", name: "Slit-scan block", set: {
    mode: "tumble", autoMorph: false, morph: 2, history: 150, depth: 1.3, solidShade: 0.12, glow: 0.45 } },
  { id: "night", name: "Night lines", set: {
    background: "#0a0a0d", wireInk: 0, wireContrast: 1.7, wireOpacity: 0.85, glow: 0.9, glowThreshold: 0.45,
    sliceClear: 0, grain: 0.12 } },
  { id: "blueprint", name: "Blueprint", set: {
    mode: "spin", spinYaw: 14, spinPitch: -4, autoMorph: false, morph: 0, background: "#1d4596",
    wireInk: 1, wireInkColor: "#f3f6ff", wireOpacity: 0.55, wireGrid: 30, wirePlanes: 22, wireWidth: 1.2,
    glow: 0.25, glowThreshold: 0.6, grain: 0.06 } },
  { id: "longexp", name: "Long exposure", set: {
    mode: "tumble", speed: 0.6, history: 180, step: 2, depth: 2.2, autoMorph: true, wirePlanes: 60, slicePlanes: 90 } },
  // made for AI depth (they also work on brightness)
  { id: "depthPoints", name: "Depth points", set: {
    mode: "sway", pivot: "newest", speed: 1, autoMorph: false, morph: 3, relief: 1, trimFar: 0.06,
    history: 90, depth: 1.2, fadeOld: 0.55, pointPlanes: 48, pointDensity: 170, pointSize: 1.1, pointOpacity: 0.9,
    background: "#0b0c0f", glow: 0.35, glowThreshold: 0.6, glowSize: 1.2, grain: 0.06, zoom: 1.35 } },
  { id: "depthSheets", name: "Depth sheets", set: {
    mode: "sway", pivot: "newest", speed: 0.8, autoMorph: false, morph: 1, relief: 1.1, trimFar: 0.04, history: 90,
    depth: 1.2, fadeOld: 0.45, slicePlanes: 40, sliceOpacity: 0.4, sliceClear: 0.15, background: "#0b0c0f",
    glow: 0.45, glowThreshold: 0.55, grain: 0.06, zoom: 1.35 } },
  { id: "tunnel", name: "Time tunnel", set: {
    mode: "sway", speed: 0.6, autoMorph: false, morph: 1, shape: "echo", spiral: 140, history: 150, step: 2,
    depth: 2.4, fadeOld: 0.2, slicePlanes: 64, sliceOpacity: 0.32, sliceClear: 0.2, background: "#08080a",
    glow: 0.6, glowThreshold: 0.5, grain: 0.06, zoom: 1.2 } },
  { id: "staircase", name: "Stack of moments", set: {
    mode: "sway", speed: 0.7, autoMorph: false, morph: 1, shape: "stack", history: 120, step: 3, depth: 1.6,
    slicePlanes: 14, sliceOpacity: 0.85, sliceClear: 0, fadeOld: 0, background: "#eeeeee", glow: 0.2, zoom: 1.0 } },
  // along the camera path
  { id: "cameraPath", name: "Camera path", set: {
    layout: "path", pathFrom: "video", mode: "sway", pivot: "newest", speed: 0.8, autoMorph: false, morph: 3,
    relief: 1, trimFar: 0.07, history: 120, fadeOld: 0.35, pointPlanes: 60, pointDensity: 150, pointSize: 1.15,
    pointOpacity: 0.9, background: "#0b0c0f", glow: 0.3, glowThreshold: 0.6, grain: 0.05, zoom: 0.8 } },
  { id: "walkThrough", name: "Walk through time", set: {
    layout: "path", pathFrom: "walk", mode: "sway", pivot: "newest", speed: 0.7, autoMorph: false, morph: 1,
    relief: 0.7, trimFar: 0.07, history: 120, step: 3, fadeOld: 0.2, slicePlanes: 40, sliceOpacity: 0.55,
    sliceClear: 0, background: "#0b0c0f", glow: 0.35, grain: 0.05, zoom: 0.75 } },
  { id: "orbitRing", name: "Orbit ring", set: {
    layout: "path", pathFrom: "orbit", pathSpeed: 1.2, mode: "still", pivot: "centre", autoMorph: false, morph: 1,
    relief: 0.5, history: 150, step: 3, fadeOld: 0.1, slicePlanes: 30, sliceOpacity: 0.75, sliceClear: 0,
    background: "#0b0c0f", glow: 0.3, grain: 0.05, zoom: 0.55, showPath: false } },
  // with a followed object
  { id: "followOne", name: "Follow one", set: {
    isolate: true, trackLine: true, mode: "sway", pivot: "newest", speed: 0.8, autoMorph: false, morph: 3, relief: 1,
    history: 120, fadeOld: 0.25, depth: 1.6, pointPlanes: 60, pointDensity: 170, pointSize: 1.1, pointOpacity: 0.95,
    background: "#0b0c0f", glow: 0.55, glowThreshold: 0.55, grain: 0.05, zoom: 1.1 } },
  { id: "followPath", name: "Follow along the path", set: {
    layout: "path", pathFrom: "video", isolate: false, dimRest: 0.8, trackLine: true, mode: "sway", pivot: "newest",
    speed: 0.8, autoMorph: false, morph: 3, relief: 1, trimFar: 0.07, history: 120, fadeOld: 0.2, pointPlanes: 60,
    pointDensity: 150, pointOpacity: 0.9, background: "#0b0c0f", glow: 0.5, glowThreshold: 0.55, grain: 0.05,
    zoom: 0.8 } },
  { id: "motionTrails", name: "Motion trails", set: {
    mode: "sway", pivot: "newest", autoMorph: false, morph: 3, relief: 0.8, motion: 0.85, history: 120,
    depth: 1.8, fadeOld: 0.2, pointPlanes: 72, pointDensity: 150, pointOpacity: 0.9, background: "#08090b",
    glow: 0.6, glowThreshold: 0.5, grain: 0.05, zoom: 1.2 } },
];

export function presetValues(id) {
  const p = PRESETS.find((x) => x.id === id) || PRESETS[0];
  return { ...defaults(), ...p.set };
}

// Format of the output (and of the preview frame).
export const FORMATS = [
  { id: "reel", name: "Vertical 9:16", w: 1080, h: 1920 },
  { id: "portrait", name: "Portrait 4:5", w: 1080, h: 1350 },
  { id: "square", name: "Square 1:1", w: 1080, h: 1080 },
  { id: "wide", name: "Wide 16:9", w: 1920, h: 1080 },
  { id: "wide4k", name: "4K 16:9", w: 3840, h: 2160 },
];

export function hexToRgb(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec(String(hex || "").trim());
  const n = m ? parseInt(m[1], 16) : 0xeeeeee;
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}
