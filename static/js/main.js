// Time Cube - the window: controls, sources, the frame clock, mouse and keys.
import { LOOKS, lookWeights, pose } from "./camera.js";
import { exportVideo } from "./export.js";
import { defaults, FORMATS, hexToRgb, PARAMS, presetValues, PRESETS, SECTIONS } from "./presets.js";
import { distance, drawMap, madePose, measuredPose } from "./paths.js";
import { Renderer } from "./renderer.js";
import { Picker } from "./track.js";
import { CameraSource, DemoSource, isImage, isVideo, PhotoSource, VideoSource } from "./sources.js";

const $ = (id) => document.getElementById(id);
const canvas = $("view");
const STORE = "timecube.v1";

// ---- state ---------------------------------------------------------------------------------------------------
let S = defaults();              // every setting in presets.js
let presetId = "classic";
const ex = { format: "reel", fps: 30, crf: 18, sound: true, length: 15, name: "", alpha: "prores" };
const view = { yaw: 0, pitch: 0, zoom: 1, panX: 0, panY: 0 };  // your mouse moves on top of the choreography
let src = null;
let playing = true;
let sceneT = 0;                  // seconds since the start (for a video: since its In point)
let lastSlot = null;             // the last frame slot pushed into the cube
let frameNo = 0;
let exporting = false;
let recording = null;
let lastNow = performance.now();
// The open video's AI depth: made once by the server (depth.py on the GPU), then kept here whole.
// state: na (not a video) | wait (its copy is on the way) | none | running | loading | ready | error | cancelled
const depth = { state: "na", mid: null, status: null, data: null, w: 0, h: 0, fps: 0, frames: 0, error: "", built: false };
// The camera path measured from that video and its depth (server: path.py), for the lens asked for.
// state: none | loading | ready | error. zp: how far away the middle of the scene is (the path's unit).
const camPath = { state: "none", data: null, fov: 0, error: "", zp: distance(0.5) };
// The followed object (server: track.py). state: none | finding | picked | following | loading | ready | error |
// cancelled. cands: what was found or cut out in the paused frame; sel: the one picked.
const track = { state: "none", busy: "", label: "", cands: [], sel: -1, labels: [], data: null, info: null, w: 0, h: 0,
  fps: 0, frames: 0, seen: 0, done: 0, total: 0, error: "" };
// What the AI sees (server: caption.py). state: none | running | ready | error. moments: [{ time, text }].
const ai = { state: "none", moments: [], place: "", done: 0, total: 0, error: "", asking: false, answer: "", question: "" };

let renderer;
try {
  renderer = new Renderer(canvas);
} catch (e) {
  document.body.innerHTML = `<p style="padding:40px;font:15px Segoe UI">Time Cube needs WebGL2: ${e.message}</p>`;
  throw e;
}

function load() {
  try {
    const d = JSON.parse(localStorage.getItem(STORE) || "{}");
    if (d.S) S = { ...defaults(), ...d.S };
    if (d.presetId) presetId = d.presetId;
    if (d.ex) Object.assign(ex, d.ex, { name: "" });
  } catch { /* first visit, or storage blocked */ }
}

let saveTimer = 0;
function save() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    try { localStorage.setItem(STORE, JSON.stringify({ S, presetId, ex })); } catch { /* storage blocked */ }
  }, 250);
}

const fps = () => ex.fps;
const format = () => FORMATS.find((f) => f.id === ex.format) || FORMATS[0];

// ---- toast ---------------------------------------------------------------------------------------------------
let toastTimer = 0;
function toast(msg, opts = {}) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.toggle("error", !!opts.error);
  if (opts.path) {
    const b = document.createElement("button");
    b.textContent = "Show in folder";
    b.onclick = () => fetch("/api/reveal", { method: "POST", body: JSON.stringify({ path: opts.path }) });
    el.appendChild(b);
  }
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), opts.ms || (opts.path ? 9000 : opts.error ? 7000 : 3200));
}

// ---- side panel ----------------------------------------------------------------------------------------------
const ctl = {};  // id -> { root, input, val, set(v) }

function fmtVal(p, v) {
  if (p.unit === "×") return `${(+v).toFixed(2)}×`;
  const dec = p.step >= 1 ? 0 : p.step >= 0.1 ? 1 : 2;
  return `${(+v).toFixed(dec)}${p.unit && p.unit !== "×" ? (p.unit.length > 2 ? " " : "") + p.unit : ""}`;
}

function buildPanel() {
  const root = $("controls");
  let closed = {};
  try { closed = JSON.parse(localStorage.getItem(STORE + ".closed") || "{}"); } catch { /* none */ }
  for (const [secId, secName] of SECTIONS) {
    const sec = document.createElement("div");
    sec.className = "sec" + (closed[secId] ? " closed" : "");
    sec.dataset.sec = secId;
    const h = document.createElement("h3");
    h.textContent = secName;
    h.onclick = () => {
      sec.classList.toggle("closed");
      closed[secId] = sec.classList.contains("closed");
      try { localStorage.setItem(STORE + ".closed", JSON.stringify(closed)); } catch { /* blocked */ }
    };
    sec.appendChild(h);
    for (const p of PARAMS.filter((x) => x.sec === secId)) sec.appendChild(buildControl(p));
    root.appendChild(sec);
  }
}

function buildControl(p) {
  const div = document.createElement("div");
  div.className = "ctl " + (p.type || "range");
  div.dataset.id = p.id;
  const c = { root: div };
  const lab = document.createElement("div");
  lab.className = "lab";
  const name = document.createElement("span");
  name.textContent = p.label;
  name.title = "Double-click: back to the preset value";
  name.ondblclick = () => setParam(p.id, presetValues(presetId)[p.id]);
  lab.appendChild(name);
  if (!p.type || p.type === "range") {
    const val = document.createElement("span");
    val.className = "val";
    lab.appendChild(val);
    const input = document.createElement("input");
    input.type = "range";
    input.min = p.min;
    input.max = p.max;
    input.step = p.step;
    input.setAttribute("aria-label", p.label);
    input.oninput = () => setParam(p.id, +input.value, { user: true });
    div.append(lab, input);
    c.set = (v) => { input.value = v; val.textContent = fmtVal(p, v); };
    c.input = input;
  } else if (p.type === "bool") {
    const l = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.onchange = () => setParam(p.id, input.checked, { user: true });
    l.append(input, document.createTextNode(p.label));
    div.appendChild(l);
    c.set = (v) => { input.checked = !!v; };
  } else if (p.type === "choice") {
    div.appendChild(lab);
    const seg = document.createElement("div");
    seg.className = "seg";
    for (const [v, text] of p.options) {
      const b = document.createElement("button");
      b.textContent = text;
      b.dataset.v = v;
      b.onclick = () => setParam(p.id, typeof p.def === "number" ? +v : v, { user: true });
      seg.appendChild(b);
    }
    div.appendChild(seg);
    c.set = (v) => { for (const b of seg.children) b.classList.toggle("on", b.dataset.v === String(v)); };
  } else if (p.type === "depth") {
    // the AI depth button, what it is doing, and a progress bar
    const go = document.createElement("button");
    go.id = "depthGo";
    go.onclick = () => depthButton();
    const bar = document.createElement("div");
    bar.className = "bar";
    bar.id = "depthBar";
    bar.appendChild(document.createElement("div"));
    const status = document.createElement("div");
    status.className = "depth-status";
    status.id = "depthStatus";
    div.append(go, bar, status);
    c.set = () => {};
  } else if (p.type === "pathmap") {
    // the camera path from above, and what it is
    const cv = document.createElement("canvas");
    cv.id = "pathMap";
    cv.className = "path-map";
    const status = document.createElement("div");
    status.className = "depth-status";
    status.id = "pathStatus";
    div.append(cv, status);
    c.set = () => {};
  } else if (p.type === "track") {
    div.innerHTML = `
      <div class="track-find"><input type="text" id="trackText" spellcheck="false"
        placeholder="What to follow: a blue car, the man…"><button id="trackFind">Find</button></div>
      <div class="track-row"><button id="trackHere">What's in this moment</button>
        <button id="trackPick">Click it in the picture</button></div>
      <div class="chips" id="trackChips"></div>
      <button id="trackGo" class="primary">Follow it</button>
      <div class="bar" id="trackBar"><div></div></div>
      <div class="depth-status" id="trackStatus"></div>
      <button id="trackClear">Clear the track</button>`;
    div.querySelector("#trackFind").onclick = () => trackFind(div.querySelector("#trackText").value);
    div.querySelector("#trackText").onkeydown = (e) => { if (e.key === "Enter") trackFind(e.target.value); };
    div.querySelector("#trackHere").onclick = trackHere;
    div.querySelector("#trackPick").onclick = () => openPicker();
    div.querySelector("#trackGo").onclick = trackButton;
    div.querySelector("#trackClear").onclick = trackClear;
    c.set = () => {};
  } else if (p.type === "aisees") {
    div.innerHTML = `
      <dl class="facts" id="aiFacts"></dl>
      <button id="aiDescribe" class="primary">Describe the video</button>
      <div class="bar" id="aiBar"><div></div></div>
      <div class="depth-status" id="aiStatus"></div>
      <ol class="moments" id="aiMoments"></ol>
      <div class="ai-place" id="aiPlace"></div>
      <div class="track-find"><input type="text" id="aiQuestion" spellcheck="false"
        placeholder="Ask about this moment…"><button id="aiAsk">Ask</button></div>
      <div class="ai-answer" id="aiAnswer"></div>`;
    div.querySelector("#aiDescribe").onclick = aiDescribe;
    div.querySelector("#aiAsk").onclick = () => aiAsk(div.querySelector("#aiQuestion").value);
    div.querySelector("#aiQuestion").onkeydown = (e) => { if (e.key === "Enter") aiAsk(e.target.value); };
    c.set = () => {};
  } else if (p.type === "color") {
    const input = document.createElement("input");
    input.type = "color";
    input.oninput = () => setParam(p.id, input.value, { user: true });
    lab.appendChild(input);
    div.appendChild(lab);
    c.set = (v) => { input.value = v; };
  }
  if (p.hint) {
    const hint = document.createElement("div");
    hint.className = "hint";
    hint.textContent = p.hint;
    div.appendChild(hint);
  }
  ctl[p.id] = c;
  return div;
}

function refreshPanel() {
  for (const p of PARAMS) ctl[p.id]?.set(S[p.id]);
  for (const p of PARAMS.filter((x) => x.only)) ctl[p.id].root.classList.toggle("hidden", S.mode !== p.only);
  for (const p of PARAMS.filter((x) => x.show)) ctl[p.id].root.classList.toggle("hidden", !p.show(S));
  for (const [id, , show] of SECTIONS) {
    document.querySelector(`.sec[data-sec="${id}"]`)?.classList.toggle("hidden", !!show && !show(S));
  }
  ctl.photoHold.root.classList.toggle("hidden", src?.kind !== "photos");
  $("preset").value = presetId;
  showDepth();
  showPath();
  showTrack();
  showAi();
}

function setParam(id, v, opts = {}) {
  const old = S[id];
  S[id] = v;
  // Touching the look yourself takes it off the choreography.
  if (opts.user && id === "morph" && S.autoMorph && drivesMorph()) {
    S.autoMorph = false;
    toast("Morph with the moves: off. The look stays where you put it.");
  }
  if (id === "history" || id === "quality") rebuildVolume();
  if (id === "step" && old !== v) lastSlot = null;
  if (id === "photoHold" && src?.kind === "photos") { lastSlot = null; updateTransport(); }
  if (["pathFrom", "pathSpeed", "layout"].includes(id)) {
    renderer.repose(poseFor);                    // every frame in the cube gets its camera again
    if (S.layout === "path" && S.pathFrom === "video") loadPath();
  }
  if (id === "camFov" && old !== v) {
    clearTimeout(lensTimer);                     // measure again once the slider rests
    lensTimer = setTimeout(loadPath, 450);
  }
  // choosing AI depth with no relief would change nothing you can see
  if (id === "depthSrc" && v === "ai" && opts.user && depthOn() && S.relief < 0.05) S.relief = 0.8;
  refreshPanel();
  save();
}

function applyPreset(id) {
  presetId = id;
  // where the depth comes from, and the video's lens, belong to the source, not to the look
  const keep = { mirror: S.mirror, transparent: S.transparent, depthSrc: S.depthSrc, camFov: S.camFov };
  S = { ...presetValues(id), ...keep };
  if (S.layout === "path" && S.pathFrom === "video") loadPath();
  rebuildVolume();
  refreshPanel();
  save();
}

const drivesMorph = () => S.mode === "reel" || S.mode === "tumble";

// ---- the cube's look for a moment in time --------------------------------------------------------------------
// output: "bg" (over the background colour), "checker" (transparency preview) or "alpha" (transparent).
function lookState(t, seed, output = "bg") {
  const p = pose(S.mode, t, S);
  const morph = S.autoMorph && p.morph !== null ? p.morph : S.morph;
  return {
    yaw: p.yaw, pitch: p.pitch, roll: p.roll,
    dragYaw: view.yaw, dragPitch: view.pitch,
    zoom: S.zoom * view.zoom, fov: S.fov, panX: view.panX, panY: view.panY,
    morph, depth: S.depth, newestBack: S.layout === "path" ? false : S.newestBack, fadeOld: S.fadeOld, pivot: S.pivot,
    shape: S.shape, spiral: S.spiral, layout: S.layout,
    path: { fov: S.camFov, zp: camPath.zp, showPath: S.showPath },
    track: { dim: trackOn() ? (S.isolate ? 1 : S.dimRest) : 0, line: trackOn() && S.trackLine,
      color: hexToRgb(S.trackColor) },
    caption: S.captionOn ? captionAt(src?.kind === "video" ? src.in + t : t) : "",
    wire: { axis: S.wireAxis, grid: S.wireGrid, stretch: S.wireStretch, planes: S.wirePlanes, width: S.wireWidth, opacity: S.wireOpacity, ink: S.wireInk,
      inkColor: hexToRgb(S.wireInkColor), diagonals: S.wireDiagonals, wave: S.wireWave, contrast: S.wireContrast },
    slices: { planes: S.slicePlanes, opacity: S.sliceOpacity, clear: S.sliceClear },
    solid: { shade: S.solidShade },
    points: { planes: S.pointPlanes, density: S.pointDensity, size: S.pointSize, opacity: S.pointOpacity },
    relief: { ai: S.depthSrc === "ai" && depthOn(), amount: S.relief, trimFar: S.trimFar, trimDark: S.trimDark,
      motion: S.motion },
    background: hexToRgb(S.background), glow: S.glow, glowSize: S.glowSize, glowThreshold: S.glowThreshold,
    grain: S.grain, seed, output,
    tone: { brightness: S.brightness, contrast: S.contrast, saturation: S.saturation, soften: S.soften, mirror: S.mirror },
  };
}

// ---- the volume ----------------------------------------------------------------------------------------------
function volumeSize() {
  const q = +S.quality;
  const a = (src?.width || 4) / (src?.height || 3);
  let w, h;
  if (a >= 1) { w = q; h = q / a; } else { h = q; w = q * a; }
  return [Math.max(8, Math.round(w / 2) * 2), Math.max(8, Math.round(h / 2) * 2)];
}

// Give the cube the current picture and start a fresh history.
function rebuildVolume() {
  if (!src) return;
  const [w, h] = volumeSize();
  renderer.setVolume(w, h, Math.round(S.history));
  if (uploadCurrent(sceneT)) renderer.fill();
  lastSlot = null;
}

// Put the source's picture for time t into the renderer. False when there is no picture yet.
function uploadCurrent(t) {
  if (!src) return false;
  if (src.kind === "video" || src.kind === "camera") {
    if (src.el.readyState < 2) return false;
    renderer.setSource(src.el, src.el.videoWidth, src.el.videoHeight);
    setDepthFor(src.kind === "video" ? src.el.currentTime : null);
    setTrackFor(src.kind === "video" ? src.el.currentTime : null);
    const at = src.kind === "video" ? src.el.currentTime : t;
    renderer.setPose(poseFor(at), at);
  } else if (src.kind === "photos") {
    const img = src.imageAt(t, fps(), S.photoHold);
    renderer.setSource(img, img.width, img.height);
    renderer.setDepth(null);
    renderer.setMask(null);
    renderer.setPose(poseFor(t), t);
  } else {
    renderer.setSource(src.imageAt(t), src.width, src.height);
    renderer.setDepth(null);
    renderer.setMask(null);
    renderer.setPose(poseFor(t), t);
  }
  return true;
}

// ---- the followed object ------------------------------------------------------------------------------------
const trackOn = () => track.state === "ready" && track.data !== null && src?.kind === "video";

// The object at a moment of the open video: its mask and centre, or null without a track.
function trackAt(time) {
  if (!trackOn()) return null;
  const i = Math.max(0, Math.min(track.frames - 1, Math.round(time * track.fps)));
  const n = track.w * track.h, f = track.info[i];
  return { mask: { data: track.data.subarray(i * n, (i + 1) * n), width: track.w, height: track.h }, key: i,
    point: f ? [f.cx, f.cy, f.near ?? 0.5] : null };
}

function setTrackFor(time) {
  const t = time === null ? null : trackAt(time);
  renderer.setMask(t ? t.mask : null, t ? t.key : null, t ? t.point : null);
}

const picker = new Picker($("picker"), $("pickerCanvas"), $("pickerHint"));
picker.onPick = (i) => { track.sel = i; track.state = "picked"; picker.set(track.cands, i); showTrack(); };
picker.onCut = (u, v) => trackCut(u, v);
$("pickerClose").onclick = () => picker.close();
$("pickerGo").onclick = () => { picker.close(); trackButton(); };

// The paused moment the Track section works on (seconds into the video).
function trackMoment() {
  if (src?.kind !== "video") return null;
  if (playing) setPlaying(false);
  return src.el.currentTime;
}

function openPicker() {
  if (src?.kind !== "video") return;
  trackMoment();
  picker.open(src.el);
  picker.set(track.cands, track.sel);
}

async function trackAsk(what, body, busy) {
  const s = src;
  const info = await s.uploaded;
  Object.assign(track, { busy, error: "" });
  showTrack();
  try {
    const j = await getJSON(`/api/track/${what}`, { id: info.id, ...body });
    if (src !== s) return null;
    return j;
  } catch (e) {
    if (src === s) { track.error = e.message; track.state = track.state === "ready" ? "ready" : "error"; }
    return null;
  } finally {
    if (src === s) { track.busy = ""; showTrack(); }
  }
}

async function trackFind(text) {
  text = (text || "").trim();
  if (!text || src?.kind !== "video") return;
  const time = trackMoment();
  openPicker();
  const j = await trackAsk("find", { time, text }, `Looking for “${text}”…`);
  if (!j) return;
  track.cands = j.candidates || [];
  track.sel = track.cands.length ? 0 : -1;
  if (track.state !== "ready") track.state = track.cands.length ? "picked" : "none";
  if (!track.cands.length) track.error = `“${text}” wasn't found here. Try other words, or click it in the picture.`;
  picker.set(track.cands, track.sel);
  showTrack();
}

async function trackHere() {
  if (src?.kind !== "video") return;
  const time = trackMoment();
  const j = await trackAsk("here", { time }, "Looking at this moment…");
  if (!j) return;
  track.labels = j.labels || [];
  if (!track.labels.length) track.error = "Nothing everyday was recognised here. Click it in the picture instead.";
  showTrack();
}

async function trackCut(u, v) {
  const time = trackMoment();
  const j = await trackAsk("cut", { time, point: [u, v] }, "Cutting it out…");
  if (!j || !j.candidates?.length) return;
  track.cands = [...track.cands.filter((c) => c.label !== "your pick"), j.candidates[0]];
  track.sel = track.cands.length - 1;
  if (track.state !== "ready") track.state = "picked";
  picker.set(track.cands, track.sel);
  showTrack();
}

let trackPoll = 0;
async function trackButton() {
  if (src?.kind !== "video") return;
  const s = src, info = await s.uploaded;
  if (track.state === "following") {
    trackStatus(s, await getJSON("/api/track/cancel", { id: info.id }).catch((e) => ({ state: "error", error: e.message })));
    return;
  }
  const c = track.cands[track.sel];
  if (!c) { openPicker(); return; }
  const time = trackMoment();
  // the middle of the box as well as the box: SAM then knows which part is meant
  const point = [(c.box[0] + c.box[2]) / 2, (c.box[1] + c.box[3]) / 2];
  track.label = c.label || "it";
  try {
    trackStatus(s, await getJSON("/api/track/follow", { id: info.id, time, box: c.box, outline: c.outline,
      point: c.label === "your pick" ? point : null, label: track.label }));
  } catch (e) {
    Object.assign(track, { error: e.message });
    showTrack();
  }
}

function trackStatus(s, st) {
  if (src !== s) return;
  clearTimeout(trackPoll);
  if (st.state === "ready") { loadTrack(s, true); return; }
  if (st.state === "running") {
    Object.assign(track, { state: "following", done: st.done || 0, total: st.total || 0, label: st.label || track.label });
    trackPoll = setTimeout(async () => {
      try { trackStatus(s, await getJSON(`/api/track/status?id=${s.info.id}`)); }
      catch (e) { if (src === s) { Object.assign(track, { state: "error", error: e.message }); showTrack(); } }
    }, 400);
  } else if (st.state === "error") {
    Object.assign(track, { state: "error", error: st.error || "tracking failed" });
    toast(`Tracking failed: ${track.error}`, { error: true });
  } else if (st.state === "cancelled") {
    Object.assign(track, { state: track.cands.length ? "picked" : "none", error: "Stopped." });
  }
  showTrack();
}

async function loadTrack(s, fresh) {
  track.state = "loading";
  showTrack();
  try {
    const id = s.info.id;
    const [r, info] = await Promise.all([fetch(`/api/track/data?id=${id}`), getJSON(`/api/track/info?id=${id}`)]);
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `server error ${r.status}`);
    const data = new Uint8Array(await r.arrayBuffer());
    if (src !== s) return;
    Object.assign(track, { state: "ready", data, info: info.info, w: info.width, h: info.height, fps: info.fps,
      frames: info.frames, seen: info.seen, label: info.label, seconds: info.seconds, error: "" });
  } catch (e) {
    if (src !== s) return;
    Object.assign(track, { state: "error", data: null, error: e.message });
    showTrack();
    return;
  }
  if (fresh) {
    if (!S.isolate && S.dimRest < 0.05 && !S.trackLine) S.trackLine = true;
    toast(`Following “${track.label}”: seen in ${track.seen} of ${track.frames} frames. Try Isolate tracked, or the Follow one preset.`);
    picker.close();
  }
  lastSlot = null;      // the history starts again, every frame with the object
  feed(sceneT);
  refreshPanel();
}

async function trackCheck(s) {
  clearTimeout(trackPoll);
  Object.assign(track, { state: "none", busy: "", label: "", cands: [], sel: -1, labels: [], data: null, info: null,
    error: "", done: 0, total: 0 });
  picker.close();
  showTrack();
  if (s.kind !== "video") return;
  try {
    const info = await s.uploaded;
    if (src !== s) return;
    const st = await getJSON(`/api/track/status?id=${info.id}`);
    if (st.state === "ready") loadTrack(s, false);
    else if (st.state === "running") trackStatus(s, st);
  } catch { /* the video copy failed: the Depth section says so */ }
}

// ---- what the AI sees ---------------------------------------------------------------------------------------
// The caption of the moment at a time of the video: the last described moment before it (the first one before
// the first).
function captionAt(time) {
  if (src?.kind !== "video" || ai.state !== "ready" || !ai.moments.length) return "";
  let m = ai.moments[0];
  for (const x of ai.moments) if (x.time <= time + 0.05) m = x;
  return m.text;
}

let aiPoll = 0;
async function aiCheck(s) {
  clearTimeout(aiPoll);
  Object.assign(ai, { state: "none", moments: [], place: "", error: "", answer: "", question: "", asking: false });
  showAi();
  if (s.kind !== "video") return;
  try {
    const info = await s.uploaded;
    if (src !== s) return;
    aiStatus(s, await getJSON(`/api/caption/status?id=${info.id}`));
  } catch { /* the Depth section says what went wrong with the video copy */ }
  showAi();
}

function aiStatus(s, st) {
  if (src !== s) return;
  clearTimeout(aiPoll);
  if (st.state === "ready") {
    Object.assign(ai, { state: "ready", moments: st.moments || [], place: st.place || "", error: "" });
  } else if (st.state === "running") {
    Object.assign(ai, { state: "running", done: st.done || 0, total: st.total || 0 });
    aiPoll = setTimeout(async () => {
      try { aiStatus(s, await getJSON(`/api/caption/status?id=${s.info.id}`)); }
      catch (e) { if (src === s) { Object.assign(ai, { state: "error", error: e.message }); showAi(); } }
    }, 500);
  } else if (st.state === "error") {
    Object.assign(ai, { state: "error", error: st.error || "describing failed" });
  } else {
    Object.assign(ai, { state: "none" });
  }
  showAi();
}

async function aiDescribe() {
  if (src?.kind !== "video") return;
  const s = src, info = await s.uploaded;
  try {
    aiStatus(s, await getJSON("/api/caption/describe", { id: info.id }));
  } catch (e) {
    Object.assign(ai, { state: "error", error: e.message });
    showAi();
  }
}

async function aiAsk(question) {
  question = (question || "").trim();
  if (!question || src?.kind !== "video" || ai.asking) return;
  const s = src, info = await s.uploaded;
  if (playing) setPlaying(false);
  Object.assign(ai, { asking: true, question, answer: "" });
  showAi();
  try {
    const j = await getJSON("/api/caption/ask", { id: info.id, time: s.el.currentTime, question });
    if (src === s) ai.answer = j.answer || "(no answer)";
  } catch (e) {
    if (src === s) ai.answer = `Couldn't answer: ${e.message}`;
  }
  if (src === s) { ai.asking = false; showAi(); }
}

function showAi() {
  const status = $("aiStatus");
  if (!status) return;
  const video = src?.kind === "video";
  const facts = $("aiFacts");
  facts.replaceChildren();
  if (video) {
    const i = src.info || {};
    const mb = src.file?.size ? `${(src.file.size / 1048576).toFixed(1)} MB` : "";
    for (const [k, v] of [["File", src.file?.name], ["Picture", i.width ? `${i.width} × ${i.height}${i.codec ? ` · ${i.codec}` : ""}` : ""],
      ["Length", src.duration ? fmtTime(src.duration) : ""], ["Frame rate", i.fps ? `${i.fps} fps` : ""], ["Size", mb],
      ["Sound", i.audio === undefined ? "" : i.audio ? "yes" : "none"]]) {
      if (!v) continue;
      const dt = document.createElement("dt"), dd = document.createElement("dd");
      dt.textContent = k;
      dd.textContent = v;
      facts.append(dt, dd);
    }
  }
  const list = $("aiMoments");
  list.replaceChildren();
  for (const m of ai.state === "ready" ? ai.moments : []) {
    const li = document.createElement("li"), b = document.createElement("button");
    b.className = "t";
    b.textContent = `${m.time.toFixed(1)} s`;
    b.title = "Go there";
    b.onclick = () => { if (src?.kind === "video") { setPlaying(false); src.el.currentTime = m.time; } };
    li.append(b, document.createTextNode(` ${m.text}`));
    list.appendChild(li);
  }
  $("aiPlace").textContent = ai.state === "ready" && ai.place ? `Place, a guess: ${ai.place}` : "";
  let text = "", pct = -1;
  if (!video) text = "What the AI sees is for videos: open one.";
  else if (ai.state === "running") {
    text = ai.done ? `Describing… ${ai.done} / ${ai.total}` : "Loading the model (the first time takes about 20 s)…";
    pct = ai.total ? ai.done / ai.total : 0;
  } else if (ai.state === "error") text = ai.error;
  else if (ai.state === "none") text = "Qwen3-VL 4B looks at moments through the video and says what it sees.";
  status.textContent = text;
  status.classList.toggle("error", ai.state === "error");
  $("aiBar").hidden = pct < 0;
  $("aiBar").firstChild.style.width = `${Math.max(0, pct) * 100}%`;
  $("aiDescribe").hidden = !video || ai.state === "running";
  $("aiDescribe").textContent = ai.state === "ready" ? "Describe it again" : "Describe the video";
  for (const id of ["aiQuestion", "aiAsk"]) $(id).disabled = !video || ai.asking || ai.state === "running";
  $("aiAnswer").textContent = ai.asking ? `“${ai.question}” … (the first answer loads the model)` :
    ai.answer ? `“${ai.question}”: ${ai.answer}` : "";
}

async function trackClear() {
  if (src?.kind !== "video") return;
  const s = src, info = await s.uploaded;
  await getJSON("/api/track/clear", { id: info.id }).catch(() => {});
  if (src !== s) return;
  Object.assign(track, { state: "none", data: null, info: null, cands: [], sel: -1, labels: [], error: "" });
  picker.set([], -1);
  lastSlot = null;
  feed(sceneT);
  refreshPanel();
}

function showTrack() {
  const status = $("trackStatus");
  if (!status) return;
  const video = src?.kind === "video";
  const chips = $("trackChips");
  chips.replaceChildren();
  const chip = (text, on, click) => {
    const b = document.createElement("button");
    b.className = "chip" + (on ? " on" : "");
    b.textContent = text;
    b.onclick = click;
    chips.appendChild(b);
  };
  track.cands.forEach((c, i) => chip(`${i + 1}. ${c.label}`, i === track.sel, () => {
    track.sel = i;
    if (track.state !== "ready") track.state = "picked";
    picker.set(track.cands, i);
    showTrack();
  }));
  for (const l of track.labels) {
    chip(l.label, false, () => { $("trackText").value = l.label; trackFind(l.label); });
  }
  let text = "", button = "Follow it", pct = -1;
  const following = track.state === "following";
  if (!video) text = "Tracking is for videos: open one.";
  else if (track.busy) text = track.busy;
  else if (following) {
    text = `Following “${track.label}”… ${track.done} / ${track.total || "?"}`;
    pct = track.total ? track.done / track.total : 0;
    button = "Stop";
  } else if (track.state === "loading") text = "Loading the track…";
  else if (track.state === "ready") {
    text = `Following “${track.label}”: seen in ${track.seen} of ${track.frames} frames.` +
      (track.cands.length ? " Pick another and Follow it to change it." : "");
  } else if (track.state === "picked") text = `${track.cands.length} found: pick one, then Follow it.`;
  else text = "Pause where the thing is in clear view. Type it and press Find, or click it in the picture.";
  if (track.error && !following) text = `${track.error} ${track.state === "ready" ? text : ""}`.trim();
  status.textContent = text;
  status.classList.toggle("error", !!track.error && !following && track.error !== "Stopped.");
  for (const id of ["trackFind", "trackHere", "trackPick", "trackText"]) $(id).disabled = !video || following || !!track.busy;
  $("trackGo").textContent = button;
  $("trackGo").disabled = !video || !!track.busy || (!following && track.sel < 0);
  $("trackGo").classList.toggle("primary", !following);
  $("trackBar").hidden = pct < 0;
  $("trackBar").firstChild.style.width = `${Math.max(0, pct) * 100}%`;
  $("trackClear").hidden = track.state !== "ready";
  $("pickerGo").disabled = track.sel < 0 || following;
  $("pickerHint").textContent = track.busy || (track.cands.length
    ? "Click an outline to pick it, or click anything else to cut it out. Then Follow it."
    : "Click the thing you want to follow.");
}

// ---- the camera path ----------------------------------------------------------------------------------------
// Where the camera was at a moment (seconds into the source; for a video, its own time), or null (where it started).
function poseFor(time) {
  if (S.pathFrom === "video") return camPath.state === "ready" && depthOn() ? measuredPose(camPath.data, time) : null;
  return madePose(S.pathFrom, time * S.pathSpeed, camPath.zp);
}

let lensTimer = 0;
// The open video's measured path, for the lens in the panel (the server measures it if it hasn't yet).
async function loadPath() {
  if (!depthOn() || !depth.mid) return;
  const fov = Math.round(S.camFov), s = src, mid = depth.mid;
  if (camPath.state === "ready" && camPath.fov === fov && camPath.mid === mid) return;
  if (camPath.state === "loading" && camPath.fov === fov && camPath.mid === mid) return;
  Object.assign(camPath, { state: "loading", fov, mid, error: "" });
  showPath();
  try {
    const j = await getJSON(`/api/depth/path?id=${mid}&fov=${fov}`);
    if (src !== s || camPath.fov !== fov) return;
    Object.assign(camPath, { state: "ready", data: { fps: j.fps, frames: j.frames, poses: new Float32Array(j.poses.flat()) },
      steps: j.steps, solved: j.solved, seconds: j.seconds });
  } catch (e) {
    if (src !== s || camPath.fov !== fov) return;
    Object.assign(camPath, { state: "error", data: null, error: e.message });
  }
  renderer.repose(poseFor);
  showPath();
}

// How far away the subject is: the median nearness of the middle of the picture (where the subject usually is),
// over the whole video. The camera path view turns around that distance, and the made-up paths are sized by it.
function sceneMiddle() {
  if (!depth.data) return distance(0.5);
  const { w, h, frames } = depth, vals = [];
  const fStep = Math.max(1, Math.floor(frames / 40));
  for (let f = 0; f < frames; f += fStep) {
    for (let y = Math.floor(h * 0.25); y < h * 0.75; y += 4) {
      for (let x = Math.floor(w * 0.3); x < w * 0.7; x += 4) vals.push(depth.data[(f * h + y) * w + x]);
    }
  }
  vals.sort((a, b) => a - b);
  return distance(vals[vals.length >> 1] / 255);
}

// The Camera path section: the map and what the path is.
function showPath() {
  const status = $("pathStatus");
  if (!status) return;
  let text = "";
  if (S.pathFrom === "video") {
    if (src?.kind !== "video") text = "From video is measured from a video. Open one, or choose Orbit, Walk or Turn.";
    else if (!depthOn()) text = "From video needs the video's AI depth: build it in Depth below. Until then the camera stays put.";
    else if (camPath.state === "loading") text = "Measuring the camera path…";
    else if (camPath.state === "error") text = `The camera path: ${camPath.error}`;
    else if (camPath.state === "ready") {
      text = `Measured from the video in ${camPath.seconds} s: ${camPath.solved} of ${camPath.steps} steps solved` +
        ` (lens ${camPath.fov}°).`;
    }
  } else {
    text = { orbit: "Orbit: a made-up path round the middle of the scene.",
      walk: "Walk: a made-up path straight ahead.", turn: "Turn: the camera turns on the spot (made up)." }[S.pathFrom];
  }
  status.textContent = text;
  status.classList.toggle("error", S.pathFrom === "video" && camPath.state === "error");
  drawPathMap();
}

function drawPathMap() {
  const cv = $("pathMap");
  if (!cv || S.layout !== "path") return;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const [w, h] = [Math.round(cv.clientWidth * dpr), Math.round(cv.clientHeight * dpr)];
  if (!w || !h) return;
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  // the whole path, sampled ten times a second (a measured one: every frame)
  const track = [];
  const add = (m) => track.push({ x: m[12], z: -m[14], hx: -m[8], hz: m[10] });
  let rate = 10;
  if (S.pathFrom === "video" && camPath.state === "ready" && depthOn()) {
    rate = camPath.data.fps;
    for (let i = 0; i < camPath.data.frames; i++) add(camPath.data.poses.subarray(i * 16, i * 16 + 16));
  } else {
    const len = src?.kind === "video" ? src.duration : src?.kind === "photos" ? src.length(fps(), S.photoHold) : 20;
    for (let t = 0; t <= Math.max(1, len); t += 1 / rate) add(poseFor(t) || [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  }
  // the stretch of time in the cube now
  const times = renderer.layerTimes || [0];
  let lo = Infinity, hi = -Infinity;
  for (const t of times) { lo = Math.min(lo, t); hi = Math.max(hi, t); }
  const wrap = (t) => (src?.kind === "video" || track.length < 2 ? t : t % (track.length / rate));
  drawMap(cv, track, [Math.round(wrap(lo) * rate), Math.round(wrap(hi) * rate)]);
}

// ---- AI depth ------------------------------------------------------------------------------------------------
const depthOn = () => depth.state === "ready" && src?.kind === "video" && depth.data !== null;

// The depth frame for a moment of the open video (seconds from its start), or null.
function depthAt(time) {
  if (!depthOn()) return null;
  const i = Math.max(0, Math.min(depth.frames - 1, Math.round(time * depth.fps)));
  const n = depth.w * depth.h;
  return { data: depth.data.subarray(i * n, (i + 1) * n), width: depth.w, height: depth.h, index: i };
}

function setDepthFor(time) {
  const d = time === null ? null : depthAt(time);
  renderer.setDepth(d, d ? d.index : null);
}

async function getJSON(url, body) {
  const r = await fetch(url, body === undefined ? undefined : { method: "POST", body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok || j.error) throw new Error(j.error || `server error ${r.status}`);
  return j;
}

// A source was opened. For a video: once its copy is on the server, see whether its depth is there already.
async function depthCheck(s) {
  clearTimeout(depthPoll);
  Object.assign(camPath, { state: "none", data: null, error: "", fov: 0, mid: null, zp: distance(0.5) });
  Object.assign(depth, { state: s.kind === "video" ? "wait" : "na", mid: null, status: null, data: null, error: "",
    built: false });
  showDepth();
  if (s.kind !== "video") return;
  try {
    const info = await s.uploaded;
    if (src !== s) return;
    depth.mid = info.id;
    depthStatus(s, await getJSON(`/api/depth/status?id=${info.id}`));
  } catch (e) {
    if (src !== s) return;
    Object.assign(depth, { state: "error", error: e.message });
    showDepth();
  }
}

let depthPoll = 0;
function depthStatus(s, st) {
  if (src !== s) return;
  depth.status = st;
  clearTimeout(depthPoll);
  if (st.state === "ready") { loadDepth(s); return; }
  depth.state = st.state;
  depth.error = st.error || "";
  if (st.state === "running") {
    depthPoll = setTimeout(async () => {
      try { depthStatus(s, await getJSON(`/api/depth/status?id=${depth.mid}`)); }
      catch (e) { if (src === s) { Object.assign(depth, { state: "error", error: e.message }); showDepth(); } }
    }, 350);
  }
  if (st.state === "error" && depth.built) toast(`AI depth failed: ${depth.error}`, { error: true });
  showDepth();
}

async function loadDepth(s) {
  depth.state = "loading";
  showDepth();
  try {
    const r = await fetch(`/api/depth/data?id=${depth.mid}`);
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `server error ${r.status}`);
    const data = new Uint8Array(await r.arrayBuffer());
    if (src !== s) return;
    const [w, h, frames, rate] = ["X-Width", "X-Height", "X-Frames", "X-Fps"].map((k) => +r.headers.get(k));
    if (data.length !== w * h * frames) throw new Error("the depth came back incomplete");
    Object.assign(depth, { data, w, h, frames, fps: rate, state: "ready" });
  } catch (e) {
    if (src !== s) return;
    Object.assign(depth, { state: "error", data: null, error: e.message });
    showDepth();
    return;
  }
  if (depth.built) {
    // you asked for it: show it
    S.depthSrc = "ai";
    if (S.relief < 0.05) S.relief = 0.8;
    save();
    toast("AI depth ready: near things come forward. Try the Points look (4).");
  }
  camPath.zp = sceneMiddle();
  lastSlot = null;            // the history starts again, every frame with its depth
  feed(sceneT);
  refreshPanel();
  loadPath();
}

async function depthButton() {
  if (!depth.mid || src?.kind !== "video") return;
  const s = src;
  try {
    if (depth.state === "running") {
      depthStatus(s, await getJSON("/api/depth/cancel", { id: depth.mid }));
      return;
    }
    depth.built = true;
    depthStatus(s, await getJSON("/api/depth/start", { id: depth.mid }));
  } catch (e) {
    Object.assign(depth, { state: "error", error: e.message });
    showDepth();
  }
}

// The Depth section and the progress chip on the picture follow depth.state.
function showDepth() {
  const go = $("depthGo"), status = $("depthStatus"), bar = $("depthBar"), chip = $("depthChip");
  if (!go) return;
  const st = depth.status || {};
  let text = "", button = "", pct = -1, chipText = "";
  switch (depth.state) {
    case "na":
      text = "AI depth is made for videos. The webcam, photos and the demo use brightness.";
      break;
    case "wait":
      text = "Getting the video ready…";
      button = "Build AI depth";
      break;
    case "none":
    case "cancelled":
      button = "Build AI depth";
      text = `${depth.state === "cancelled" ? "Stopped. " : ""}About ${st.estimate ? `${st.estimate} s` : "a minute"} on the ` +
        `graphics card for ${st.frames || "its"} frames. Made once: the video has it the next time you open it.`;
      break;
    case "running":
      button = "Stop";
      pct = st.total ? st.done / st.total : 0;
      text = st.done ? `Giving every frame depth · ${st.done} / ${st.total}${st.ms ? ` · ${Math.round(st.ms)} ms a frame` : ""}`
        : "Loading the depth model…";
      chipText = st.done ? `Giving every frame depth · ${st.done} / ${st.total}` : "Loading the depth model…";
      break;
    case "loading":
      text = "Loading the depth…";
      pct = 1;
      break;
    case "ready":
      text = `AI depth ready · ${depth.frames} frames at ${+depth.fps.toFixed(2)} fps.` +
        (S.depthSrc !== "ai" ? " Choose AI depth below to use it." : S.relief < 0.05 ? " Raise the Relief to see it." : "");
      break;
    case "error":
      button = depth.mid ? "Try again" : "";
      text = `AI depth: ${depth.error}`;
      break;
  }
  go.hidden = !button;
  go.textContent = button;
  go.disabled = depth.state === "wait";
  go.classList.toggle("primary", button !== "Stop");
  status.textContent = text;
  status.classList.toggle("error", depth.state === "error");
  bar.hidden = pct < 0;
  bar.firstChild.style.width = `${Math.max(0, pct) * 100}%`;
  chip.hidden = !chipText;
  chip.textContent = chipText;
}

// Push the frame for time t into the cube: one new layer per frame slot (fps / step slots a second).
function feed(t, uploaded = false) {
  const slot = Math.floor((t * fps()) / S.step + 1e-6);
  if (lastSlot === null || slot < lastSlot) {
    if (uploaded || uploadCurrent(t)) { renderer.fill(); lastSlot = slot; }
    return;
  }
  const n = Math.min(slot - lastSlot, renderer.volN);
  if (n <= 0) return;
  if (!uploaded && !uploadCurrent(t)) return;
  for (let i = 0; i < n; i++) renderer.push();
  lastSlot = slot;
}

// ---- sources -------------------------------------------------------------------------------------------------
async function useSource(next, label) {
  const old = src;
  try {
    toast(`Opening ${label}…`, { ms: 60000 });
    await next.open();
  } catch (e) {
    next.stop?.();
    toast(e.message || String(e), { error: true });
    return false;
  }
  if (old) old.stop();
  src = next;
  sceneT = 0;
  lastSlot = null;
  view.yaw = view.pitch = view.panX = view.panY = 0;
  view.zoom = 1;
  S.mirror = src.kind === "camera";
  for (const b of document.querySelectorAll(".sources button")) b.classList.remove("on");
  $({ video: "openVideo", camera: "openCamera", photos: "openPhotos", demo: "openDemo" }[src.kind]).classList.add("on");
  $("sourceName").textContent = src.kind === "video" ? `${src.file.name} · ${src.width}×${src.height} · ${fmtTime(src.duration)}`
    : src.kind === "photos" ? `${src.images.length} photo${src.images.length > 1 ? "s" : ""}`
    : src.kind === "camera" ? `Webcam · ${src.width}×${src.height}` : "Demo · drop a video to use your own";
  if (src.kind === "video") {
    const el = src.el;
    el.muted = muted;
    const onFrame = (_now, meta) => {
      if (src?.el !== el) return;
      if (!exporting) {
        const t = meta.mediaTime;
        if (!el.paused && t >= src.out - 0.5 / fps()) {
          el.currentTime = src.in;
        } else {
          sceneT = Math.max(0, t - src.in);
          renderer.setSource(el, el.videoWidth, el.videoHeight);
          setDepthFor(t);
          setTrackFor(t);
          renderer.setPose(poseFor(t), t);
          feed(sceneT, true);
        }
      }
      el.requestVideoFrameCallback(onFrame);
    };
    el.requestVideoFrameCallback(onFrame);
    el.addEventListener("ended", () => { if (!exporting && playing) { el.currentTime = src.in; el.play().catch(() => {}); } });
    el.addEventListener("seeked", () => {
      if (src?.el !== el || exporting) return;
      sceneT = Math.max(0, el.currentTime - src.in);
      lastSlot = null;
      feed(sceneT);
    });
    src.uploaded.catch((e) => toast(`Export won't be possible: ${e.message}`, { error: true }));
  }
  depthCheck(src);
  trackCheck(src);
  aiCheck(src);
  rebuildVolume();
  setPlaying(true);
  refreshPanel();
  updateTransport();
  toast(`${label[0].toUpperCase()}${label.slice(1)} ready`);
  return true;
}

function openFiles(files) {
  files = [...files];
  const vid = files.find(isVideo);
  if (vid) return useSource(new VideoSource(vid), vid.name);
  const imgs = files.filter(isImage);
  if (imgs.length) return useSource(new PhotoSource(imgs), imgs.length === 1 ? imgs[0].name : `${imgs.length} photos`);
  toast("That isn't a video or a picture", { error: true });
}

// ---- transport -----------------------------------------------------------------------------------------------
let muted = false;
const fmtTime = (t) => {
  t = Math.max(0, t || 0);
  const m = Math.floor(t / 60), s = t - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
};

function seqLength() {
  if (!src) return 0;
  if (src.kind === "video") return src.duration;
  if (src.kind === "photos") return src.length(fps(), S.photoHold);
  return 0;
}

function setPlaying(on) {
  playing = on;
  if (src?.kind === "video") {
    if (on) src.el.play().catch(() => {
      // Sound needs a click first: play silently instead.
      src.el.muted = muted = true;
      $("mute").textContent = "🔇";
      src.el.play().catch(() => {});
    });
    else src.el.pause();
  }
  $("play").textContent = on ? "❚❚" : "▶";
}

function restart() {
  if (src?.kind === "video") {
    src.el.currentTime = src.in;
  } else {
    sceneT = 0;
    lastSlot = null;
    feed(0);
  }
}

function updateTransport() {
  const len = seqLength();
  const tr = $("transport");
  tr.classList.toggle("no-scrub", !(len > 0));
  $("scrub").max = len || 1;
  $("dur").textContent = len > 0 ? fmtTime(len) : "";
  const isVid = src?.kind === "video";
  for (const id of ["setIn", "setOut", "clearRange", "mute"]) $(id).style.visibility = isVid ? "" : "hidden";
  const mark = $("rangeMark");
  if (isVid && len > 0 && (src.in > 0 || src.out < len - 1e-3)) {
    mark.style.display = "";
    mark.style.left = `${(src.in / len) * 100}%`;
    mark.style.width = `${((src.out - src.in) / len) * 100}%`;
  } else {
    mark.style.display = "none";
  }
  $("export").disabled = src?.kind === "camera";
  $("export").title = src?.kind === "camera" ? "The webcam is live: use Record" : "Render a video frame by frame (E)";
}

function updateClock() {
  if (!src) return;
  let t = sceneT, len = seqLength();
  if (src.kind === "video") t = src.el.currentTime;
  else if (src.kind === "photos" && len > 0) t = sceneT % len;
  $("time").textContent = fmtTime(t);
  if (len > 0 && document.activeElement !== $("scrub")) $("scrub").value = t;
}

// ---- layout --------------------------------------------------------------------------------------------------
const MAX_LIVE = 1920;  // longest side of a live recording
let cssW = 0, cssH = 0;
function layout() {
  const f = format();
  const st = $("stage").getBoundingClientRect();
  const pad = 22;
  const scale = Math.max(0.05, Math.min((st.width - pad * 2) / f.w, (st.height - pad * 2) / f.h));
  cssW = Math.max(40, Math.floor(f.w * scale));
  cssH = Math.max(40, Math.floor(f.h * scale));
  canvas.style.width = `${cssW}px`;
  canvas.style.height = `${cssH}px`;
  let bw, bh;
  if (recording) {
    // record at full size, but live encoding can't keep up beyond about 1080p: 4K is for Export MP4
    const k = Math.min(1, MAX_LIVE / Math.max(f.w, f.h));
    [bw, bh] = [Math.round((f.w * k) / 2) * 2, Math.round((f.h * k) / 2) * 2];
  } else {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const k = Math.min(1, f.w / (cssW * dpr), 2048 / Math.max(cssW * dpr, cssH * dpr));
    bw = Math.round(cssW * dpr * k);
    bh = Math.round(cssH * dpr * k);
  }
  if (canvas.width !== bw || canvas.height !== bh) {
    canvas.width = bw;
    canvas.height = bh;
  }
}
new ResizeObserver(layout).observe($("stage"));

// ---- main loop -----------------------------------------------------------------------------------------------
function tick(now) {
  const dt = Math.min(0.1, (now - lastNow) / 1000);
  lastNow = now;
  if (src && !exporting) {
    if (src.kind !== "video" && playing) {
      sceneT += dt;
      feed(sceneT);
    }
    const s = lookState(sceneT, frameNo++, S.transparent && !recording ? "checker" : "bg");
    renderer.render(s, canvas.width, canvas.height, true);
    showLook(s.morph);
    if (frameNo % 6 === 0) updateClock();
    if (frameNo % 8 === 0 && S.layout === "path") drawPathMap();
  }
  requestAnimationFrame(tick);
}

let shownLook = -1;
function showLook(m) {
  const w = lookWeights(m);
  const ws = [w.wire, w.slices, w.solid, w.points];
  const best = ws.indexOf(Math.max(...ws));
  const driven = S.autoMorph && drivesMorph();
  ctl.morph.root.classList.toggle("driven", driven);
  if (driven) ctl.morph.set(Math.round(m * 100) / 100);
  if (best !== shownLook) {
    shownLook = best;
    for (const b of document.querySelectorAll(".looks button")) b.classList.toggle("on", +b.dataset.look === best);
  }
  $("lookChip").textContent = `${{ reel: "Tour", tumble: "Tumble", sway: "Sway", spin: "Spin", still: "Still" }[S.mode]} · ${LOOKS[best]}` +
    (S.layout === "path" ? ` · along the camera path (${{ video: "from video", orbit: "orbit", walk: "walk", turn: "turn" }[S.pathFrom]})`
      : `${S.shape !== "cube" ? ` · ${S.shape[0].toUpperCase()}${S.shape.slice(1)}` : ""}${Math.abs(S.spiral) >= 1 ? ` · spiral ${S.spiral}°` : ""}`) +
    `${S.relief > 0.001 ? ` · ${S.depthSrc === "ai" && depthOn() ? "AI depth" : "brightness depth"}` : ""}` +
    `${trackOn() ? ` · following “${track.label}”` : ""}` +
    `${S.transparent ? " · transparent" : ""}`;
}

// ---- mouse: turn, pan, zoom ----------------------------------------------------------------------------------
let drag = null;
canvas.addEventListener("pointerdown", (e) => {
  drag = { x: e.clientX, y: e.clientY, pan: e.shiftKey || e.button === 1 || e.button === 2 };
  canvas.setPointerCapture(e.pointerId);
  canvas.classList.add("dragging");
});
canvas.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX;
  drag.y = e.clientY;
  if (drag.pan) {
    view.panX += (dx / cssW) * 2.2;
    view.panY -= (dy / cssH) * 2.2;
  } else {
    view.yaw += dx * 0.45;
    view.pitch = Math.max(-89, Math.min(89, view.pitch + dy * 0.45));
  }
});
const endDrag = () => { drag = null; canvas.classList.remove("dragging"); };
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  view.zoom = Math.max(0.2, Math.min(5, view.zoom * Math.exp(-e.deltaY * 0.0012)));
}, { passive: false });
canvas.addEventListener("dblclick", resetView);

function resetView() {
  view.yaw = view.pitch = view.panX = view.panY = 0;
  view.zoom = 1;
}

// ---- files: buttons and drag & drop --------------------------------------------------------------------------
$("openVideo").onclick = () => $("fileVideo").click();
$("openPhotos").onclick = () => $("filePhotos").click();
$("fileVideo").onchange = (e) => { if (e.target.files.length) openFiles(e.target.files); e.target.value = ""; };
$("filePhotos").onchange = (e) => { if (e.target.files.length) openFiles(e.target.files); e.target.value = ""; };
$("openCamera").onclick = () => useSource(new CameraSource(), "the webcam");
$("openDemo").onclick = () => useSource(new DemoSource(), "the demo");

let dragDepth = 0;
window.addEventListener("dragenter", (e) => { e.preventDefault(); dragDepth++; $("drop").classList.add("show"); });
window.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; $("drop").classList.remove("show"); } });
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  $("drop").classList.remove("show");
  if (e.dataTransfer?.files?.length) openFiles(e.dataTransfer.files);
});

// ---- top bar and panel buttons -------------------------------------------------------------------------------
for (const p of PRESETS) $("preset").add(new Option(p.name, p.id));
$("preset").onchange = (e) => { applyPreset(e.target.value); toast(`Preset: ${PRESETS.find((p) => p.id === presetId).name}`); };
for (const f of FORMATS) $("format").add(new Option(`${f.name} · ${f.w}×${f.h}`, f.id));
$("format").onchange = (e) => { ex.format = e.target.value; layout(); save(); };

for (const b of document.querySelectorAll(".looks button")) {
  b.onclick = () => setParam("morph", +b.dataset.look, { user: true });
}
$("resetAll").onclick = () => { applyPreset(presetId); toast("Back to the preset"); };
$("resetView").onclick = resetView;
$("play").onclick = () => setPlaying(!playing);
$("restart").onclick = restart;
$("scrub").oninput = (e) => {
  const v = +e.target.value;
  if (src?.kind === "video") src.el.currentTime = v;
  else if (src?.kind === "photos") { sceneT = v; lastSlot = null; feed(sceneT); }
};
$("mute").onclick = () => {
  muted = !muted;
  if (src?.kind === "video") src.el.muted = muted;
  $("mute").textContent = muted ? "🔇" : "🔊";
};
$("setIn").onclick = () => {
  if (src?.kind !== "video") return;
  src.in = Math.min(src.el.currentTime, src.out - 0.2);
  updateTransport();
  toast(`Export starts at ${fmtTime(src.in)}`);
};
$("setOut").onclick = () => {
  if (src?.kind !== "video") return;
  src.out = Math.max(src.el.currentTime, src.in + 0.2);
  updateTransport();
  toast(`Export ends at ${fmtTime(src.out)}`);
};
$("clearRange").onclick = () => {
  if (src?.kind !== "video") return;
  src.in = 0;
  src.out = src.duration;
  updateTransport();
  toast("Export uses the whole clip");
};
$("help").onclick = () => $("helpDlg").showModal();

// ---- light / dark window (the cube's own background is a setting of its own) ---------------------------------
function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  $("theme").textContent = theme === "dark" ? "☀" : "☾";
  $("theme").title = theme === "dark" ? "Light window (T)" : "Dark window (T)";
  try { localStorage.setItem("timecube.theme", theme); } catch { /* storage blocked */ }
}
setTheme(document.documentElement.dataset.theme === "light" ? "light" : "dark");
$("theme").onclick = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");

// ---- PNG still -----------------------------------------------------------------------------------------------
async function saveBlob(blob, ext, name, extra = "") {
  const r = await fetch(`/api/save?ext=${ext}&name=${encodeURIComponent(name)}${extra}`, { method: "POST", body: blob });
  const j = await r.json();
  if (!r.ok || j.error) throw new Error(j.error || "saving failed");
  return j;
}

async function snapPNG() {
  if (!src || exporting) return;
  const f = format();
  renderer.render(lookState(sceneT, frameNo, S.transparent ? "alpha" : "bg"), f.w, f.h, false);
  const px = renderer.read(f.w, f.h);
  const img = new ImageData(f.w, f.h);
  const row = f.w * 4;
  for (let y = 0; y < f.h; y++) img.data.set(px.subarray((f.h - 1 - y) * row, (f.h - y) * row), y * row);
  const oc = new OffscreenCanvas(f.w, f.h);
  oc.getContext("2d").putImageData(img, 0, 0);
  const blob = await oc.convertToBlob({ type: "image/png" });
  try {
    const j = await saveBlob(blob, "png", `${src.name} time cube`);
    toast(`Saved ${j.path.split(/[\\/]/).pop()}`, { path: j.path });
  } catch (e) {
    toast(e.message, { error: true });
  }
}
$("snap").onclick = snapPNG;

// ---- live recording (what you see, mouse moves included) -----------------------------------------------------
async function toggleRecord() {
  if (recording) { recording.stop(); return; }
  if (!src || exporting || !window.MediaRecorder) return;
  const chunks = [];
  // H.264 first: Chrome encodes it fast enough to keep every frame (VP9 drops most of them at 1080p)
  const mime = ["video/webm;codecs=h264", "video/mp4;codecs=avc1", "video/webm;codecs=vp9", "video/webm;codecs=vp8",
    "video/webm"].find((m) => MediaRecorder.isTypeSupported(m));
  recording = { stop: () => {} };
  $("record").disabled = true;
  layout();                                  // the canvas goes to full size...
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));  // ...and draws at it first
  $("record").disabled = false;
  const [rw, rh] = [canvas.width, canvas.height];
  const stream = canvas.captureStream(fps());
  if (src.kind === "video" && !muted) {
    try {
      const a = src.el.captureStream().getAudioTracks()[0];
      if (a) stream.addTrack(a);
    } catch { /* record without sound */ }
  }
  const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 30e6 });
  rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  rec.onerror = (e) => toast(`Recording failed: ${e.error?.message || "the recorder stopped"}`, { error: true });
  rec.onstop = async () => {
    recording = null;
    $("record").classList.remove("recording");
    $("record").lastChild.textContent = "Record";
    layout();
    if (!chunks.length) { toast("The recording came out empty. Try again, or use Export video.", { error: true }); return; }
    toast("Saving the recording…", { ms: 60000 });
    try {
      const j = await saveBlob(new Blob(chunks, { type: mime.split(";")[0] }), "mp4", `${src?.name || "time cube"} live`,
        `&w=${rw}&h=${rh}&fps=${fps()}`);
      toast(`Saved ${j.path.split(/[\\/]/).pop()}`, { path: j.path });
    } catch (e) {
      toast(e.message, { error: true });
    }
  };
  recording = rec;
  rec.start(1000);
  $("record").classList.add("recording");
  $("record").lastChild.textContent = "Stop";
  toast(`Recording at ${rw}×${rh}${S.transparent ? " on the background colour (live recordings can't be transparent)" : ""}.` +
    " Turn the cube with the mouse; press R or Stop to finish.");
}
$("record").onclick = toggleRecord;

// ---- export dialog -------------------------------------------------------------------------------------------
let exportAbort = null;
function openExport() {
  if (!src || exporting) return;
  if (src.kind === "camera") { toast("The webcam is live: use Record instead", { error: true }); return; }
  const dlg = $("exportDlg");
  const sel = $("exFormat");
  if (!sel.options.length) for (const f of FORMATS) sel.add(new Option(`${f.name} · ${f.w}×${f.h}`, f.id));
  sel.value = ex.format;
  const isVid = src.kind === "video";
  $("exRangeRow").hidden = !isVid;
  $("exSoundRow").hidden = !isVid;
  $("exLenRow").hidden = isVid;
  if (isVid) {
    $("exRange").textContent = `${fmtTime(src.in)} – ${fmtTime(src.out)} (${(src.out - src.in).toFixed(1)} s). Set In / Out under the picture.`;
    $("exSound").checked = ex.sound;
    $("exSound").disabled = src.info ? !src.info.audio : false;
  } else {
    $("exLen").value = src.kind === "photos" ? +src.length(ex.fps, S.photoHold).toFixed(2) : ex.length;
  }
  $("exName").value = `${src.name} time cube`;
  segSet("exFps", ex.fps);
  segSet("exQuality", ex.crf);
  alphaRows();
  fetch("/api/ping").then((r) => r.json()).then((j) => { $("exWhere").textContent = `Saved to ${j.outDir}`; }).catch(() => {});
  $("exProgress").hidden = true;
  $("exGo").disabled = false;
  $("exCancel").textContent = "Close";
  dlg.showModal();
}

function segSet(id, v) {
  for (const b of $(id).children) b.classList.toggle("on", b.dataset.v === String(v));
}

const ALPHA_NOTES = {
  prores: "ProRes 4444 with alpha: for Premiere, After Effects, Resolve and Final Cut. Large files; Best is 4444 XQ.",
  webm: "VP9 with alpha in WebM: small files, transparent in Chrome, Edge and Firefox. Not for Final Cut.",
};

// The Background / Video type rows of the export dialog follow the Transparent background setting.
function alphaRows() {
  segSet("exBg", S.transparent ? "transparent" : "colour");
  segSet("exAlpha", ex.alpha);
  $("exAlphaRow").hidden = !S.transparent;
  $("exAlphaNote").hidden = !S.transparent;
  $("exAlphaNote").textContent = `${ALPHA_NOTES[ex.alpha]} Transparent frames go out uncompressed, so this export is slower.`;
}
for (const b of $("exBg").children) {
  b.onclick = () => { if (!exporting) { setParam("transparent", b.dataset.v === "transparent"); alphaRows(); } };
}
for (const b of $("exAlpha").children) {
  b.onclick = () => { if (!exporting) { ex.alpha = b.dataset.v; alphaRows(); save(); } };
}
for (const id of ["exFps", "exQuality"]) {
  for (const b of $(id).children) {
    b.onclick = () => {
      if (exporting) return;
      if (id === "exFps") ex.fps = +b.dataset.v; else ex.crf = +b.dataset.v;
      segSet(id, b.dataset.v);
      if (id === "exFps" && src?.kind === "photos") $("exLen").value = +src.length(ex.fps, S.photoHold).toFixed(2);
      save();
    };
  }
}

$("export").onclick = openExport;
$("exCancel").onclick = () => {
  if (exporting) { exportAbort?.abort(); return; }
  $("exportDlg").close();
};
$("exportDlg").addEventListener("cancel", (e) => { if (exporting) { e.preventDefault(); exportAbort?.abort(); } });
$("exGo").onclick = async () => {
  if (exporting || !src) return;
  ex.format = $("exFormat").value;
  ex.sound = $("exSound").checked;
  if (src.kind !== "video") ex.length = Math.max(0.5, +$("exLen").value || 10);
  $("format").value = ex.format;
  save();
  const f = format();
  exporting = true;
  exportAbort = new AbortController();
  $("exProgress").hidden = false;
  $("exGo").disabled = true;
  $("exCancel").textContent = "Stop";
  $("exStatus").textContent = "Starting…";
  $("exBar").style.width = "0%";
  if (src.kind === "video") src.el.pause();
  const t0 = performance.now();
  try {
    const res = await exportVideo({
      renderer, src, S, volumeSize, canvas, depthAt, poseAt: poseFor, trackAt,
      state: (t, i) => lookState(t, i),
    }, {
      w: f.w, h: f.h, fps: ex.fps, crf: ex.crf, sound: ex.sound && src.kind === "video",
      alpha: S.transparent ? ex.alpha : null,
      length: ex.length, name: $("exName").value.trim() || `${src.name} time cube`, raw: !!window.__forceRaw,
    }, (i, n) => {
      const el = (performance.now() - t0) / 1000;
      const rate = i / Math.max(el, 1e-3);
      $("exBar").style.width = `${((i / n) * 100).toFixed(1)}%`;
      $("exStatus").textContent = `Frame ${i} of ${n} · ${rate.toFixed(1)} fps · about ${fmtTime((n - i) / Math.max(rate, 0.1))} left`;
    }, exportAbort.signal);
    $("exStatus").textContent = `Done in ${fmtTime((performance.now() - t0) / 1000)}: ${res.path}`;
    $("exBar").style.width = "100%";
    toast(`Saved ${res.path.split(/[\\/]/).pop()}`, { path: res.path });
    window.__lastExport = res;
  } catch (e) {
    const stopped = e.name === "AbortError";
    $("exStatus").textContent = stopped ? "Stopped. Nothing was saved." : `Failed: ${e.message}`;
    if (!stopped) toast(e.message, { error: true });
    window.__lastExport = { error: e.message };
  } finally {
    exporting = false;
    exportAbort = null;
    $("exGo").disabled = false;
    $("exCancel").textContent = "Close";
    layout();                 // the canvas may have rendered at full size
    rebuildVolume();          // the export used the cube: start the preview fresh
    setPlaying(playing);
  }
};

// ---- keys ----------------------------------------------------------------------------------------------------
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && picker.isOpen) { picker.close(); return; }
  if (e.target.closest?.("input[type=text], input[type=number], select, dialog") || e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === " ") { e.preventDefault(); setPlaying(!playing); }
  else if (e.key === "Home") restart();
  else if (k >= "1" && k <= "4" && k.length === 1) setParam("morph", +k - 1, { user: true });
  else if (k === "v") $("openVideo").click();
  else if (k === "c") $("openCamera").click();
  else if (k === "i") $("setIn").click();
  else if (k === "o") $("setOut").click();
  else if (k === "m") $("mute").click();
  else if (k === "p") snapPNG();
  else if (k === "r") toggleRecord();
  else if (k === "e") openExport();
  else if (k === "t") $("theme").click();
  else if (k === "?" || k === "/") $("helpDlg").showModal();
});

// ---- heartbeat: the Desktop launcher closes the server a little after this window closes ---------------------
const beat = () => fetch("/api/heartbeat", { method: "POST", body: "{}" }).catch(() => {});
beat();
setInterval(beat, 20000);

// ---- start ---------------------------------------------------------------------------------------------------
load();
buildPanel();
$("format").value = ex.format;
refreshPanel();
layout();
useSource(new DemoSource(), "the demo").then(() => requestAnimationFrame(tick));

// For the end-to-end test.
window.__tc = {
  get S() { return S; }, setParam, applyPreset, openFiles, useSource, renderer, view, ex,
  get src() { return src; }, get sceneT() { return sceneT; }, setPlaying, restart, lookState, snapPNG,
  get exporting() { return exporting; }, toggleRecord, CameraSource, DemoSource, setTheme, depth, depthButton,
  depthAt, camPath, poseFor, loadPath, track, trackFind, trackHere, trackCut, trackButton, trackClear, trackAt,
  picker, ai, aiDescribe, aiAsk, captionAt,
  seek(t) {  // jump the clock (the cube restarts from that moment)
    if (src?.kind === "video") { src.el.currentTime = src.in + t; return; }
    sceneT = t;
    lastSlot = null;
    feed(t);
  },
};
