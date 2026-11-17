// Frame-by-frame MP4 export.
// A video source is read by the server (ffmpeg) at exactly the export frame rate, already shrunk to the cube's
// size; photos and the demo are drawn here for each frame's time. Every frame is rendered at full output size.
// Fast path: Chrome encodes H.264 on the GPU (WebCodecs) and only the small compressed stream goes to the server,
// which copies it into the MP4 with the video's sound. Fallback: raw pixels are read back and the server encodes.

// fetch, tried again a few times when the connection drops (the server ignores a frame it already has).
async function fetchRetry(url, init, tries = 4) {
  for (let k = 1; ; k++) {
    try {
      return await fetch(url, init);
    } catch (e) {
      if (k >= tries) throw new Error(`the local server stopped answering (${e.message})`);
      await new Promise((r) => setTimeout(r, 250 * k));
    }
  }
}

async function post(url, body) {
  const r = await fetchRetry(url, { method: "POST", body: body === undefined ? "" : body instanceof Blob ? body : JSON.stringify(body) });
  let j = {};
  try { j = await r.json(); } catch { /* empty */ }
  if (!r.ok || j.error) throw new Error(j.error || `server error ${r.status}`);
  return j;
}

// Reads decoded frames in batches, always one batch ahead.
class FrameReader {
  constructor(sid, w, h) {
    this.sid = sid;
    this.size = w * h * 4;
    this.w = w;
    this.h = h;
    this.queue = [];
    this.done = false;
    this.asked = 0;      // frames requested so far
    this.next = this.fetch();
    this.next.catch(() => {});
  }

  async fetch() {
    const from = this.asked;
    const r = await fetchRetry(`/api/decode/${this.sid}/frames?n=8&from=${from}`);
    if (!r.ok) {
      let msg = `server error ${r.status}`;
      try { msg = (await r.json()).error || msg; } catch { /* not json */ }
      throw new Error(msg);
    }
    const n = +r.headers.get("X-Frames") || 0;
    const done = r.headers.get("X-Done") === "1";
    const buf = new Uint8Array(await r.arrayBuffer());
    this.asked = from + n;
    const frames = [];
    for (let i = 0; i < n; i++) frames.push(buf.subarray(i * this.size, (i + 1) * this.size));
    return { frames, done: done || n === 0 };
  }

  // Wait for a read-ahead still on its way (before the session is closed).
  async settle() {
    try { await this.next; } catch { /* failed or closed */ }
  }

  // The next frame, or null at the end of the clip.
  async read() {
    while (!this.queue.length && !this.done) {
      const b = await this.next;
      this.queue.push(...b.frames);
      this.done = b.done;
      if (!this.done) {
        this.next = this.fetch();
        this.next.catch(() => {});  // a failure surfaces when it is awaited
      }
    }
    return this.queue.shift() || null;
  }
}

function abortError() {
  const e = new Error("stopped");
  e.name = "AbortError";
  return e;
}

// Bits per pixel per frame for the two quality levels (crf 18 = High, 14 = Best).
const BPP = { 18: 0.42, 14: 0.7 };

// A WebCodecs H.264 setup this browser can run for this size, or null.
export async function h264Config(w, h, fps, crf) {
  if (!window.VideoEncoder || !window.VideoFrame) return null;
  const bitrate = Math.round(Math.min(160e6, Math.max(4e6, w * h * fps * (BPP[crf] || BPP[18]))));
  const mbs = Math.ceil(w / 16) * Math.ceil(h / 16);
  // High profile at a level that fits the frame: 4.2 up to 1080p-sized frames, 5.1 / 5.2 for 4K.
  const codecs = mbs <= 8704 ? ["avc1.64002A", "avc1.640033"] : ["avc1.640033", "avc1.640034"];
  for (const hardwareAcceleration of ["prefer-hardware", "no-preference"]) {
    for (const codec of codecs) {
      const config = { codec, width: w, height: h, bitrate, framerate: fps, bitrateMode: "variable",
        latencyMode: "quality", hardwareAcceleration, avc: { format: "annexb" } };
      try {
        const r = await VideoEncoder.isConfigSupported(config);
        if (r.supported) return r.config;
      } catch { /* try the next one */ }
    }
  }
  return null;
}

// Encodes canvas frames with WebCodecs and streams the H.264 to the server in chunks of about 2 MB.
class H264Sink {
  constructor(sid, config, canvas, fps) {
    this.sid = sid;
    this.canvas = canvas;
    this.fps = fps;
    this.parts = [];
    this.bytes = 0;
    this.sent = 0;          // chunks sent
    this.sending = null;
    this.error = null;
    this.encoder = new VideoEncoder({
      output: (chunk) => {
        const b = new Uint8Array(chunk.byteLength);
        chunk.copyTo(b);
        this.parts.push(b);
        this.bytes += b.length;
      },
      error: (e) => { this.error = e; },
    });
    this.encoder.configure(config);
  }

  async add(i) {
    if (this.error) throw this.error;
    const us = 1e6 / this.fps;
    const frame = new VideoFrame(this.canvas, { timestamp: Math.round(i * us), duration: Math.round(us) });
    this.encoder.encode(frame, { keyFrame: i % Math.round(this.fps * 2) === 0 });
    frame.close();
    while (this.encoder.encodeQueueSize > 4) {
      await new Promise((r) => setTimeout(r, 2));
      if (this.error) throw this.error;
    }
    if (this.bytes > 2e6) await this.send();
  }

  async send() {
    if (this.sending) await this.sending;
    if (!this.parts.length) return;
    const blob = new Blob(this.parts);
    this.parts = [];
    this.bytes = 0;
    const n = this.sent++;
    this.sending = post(`/api/encode/${this.sid}/frame?i=${n}`, blob);
    this.sending.catch(() => {});
  }

  async finish(frames) {
    await this.encoder.flush();
    if (this.error) throw this.error;
    this.encoder.close();
    await this.send();
    await this.sending;
    return post(`/api/encode/${this.sid}/finish?frames=${frames}`);
  }

  async settle() {
    try { await this.sending; } catch { /* reported elsewhere */ }
  }

  close() {
    try { if (this.encoder.state !== "closed") this.encoder.close(); } catch { /* closed */ }
  }
}

// Reads the rendered pixels back and sends them raw; the server encodes (NVENC or libx264).
class RawSink {
  constructor(sid, renderer, w, h) {
    this.sid = sid;
    this.renderer = renderer;
    this.w = w;
    this.h = h;
    this.pixels = new Uint8Array(w * h * 4);
    this.sending = null;
  }

  async add(i) {
    this.renderer.read(this.w, this.h, this.pixels);
    if (this.sending) await this.sending;  // one frame on its way while the next one renders
    this.sending = post(`/api/encode/${this.sid}/frame?i=${i}`, new Blob([this.pixels]));  // Blob copies
    this.sending.catch(() => {});
  }

  async finish() {
    await this.sending;
    return post(`/api/encode/${this.sid}/finish`);
  }

  async settle() {
    try { await this.sending; } catch { /* reported elsewhere */ }
  }

  close() {}
}

// app: { renderer, src, S, volumeSize(), state(t, frameIndex), canvas, depthAt(seconds) -> AI depth frame | null,
//        poseAt(seconds) -> the camera there (camera path) | null, trackAt(seconds) -> the followed object | null }
// cfg: { w, h, fps, crf, sound, length, name, raw (force the raw path), alpha ("prores" | "webm": transparent) }
export async function exportVideo(app, cfg, onProgress, signal) {
  const { renderer, src, S, canvas } = app;
  const { w, h, fps } = cfg;
  const isVid = src.kind === "video";
  let start = 0, length = cfg.length;
  let info = null;
  if (isVid) {
    info = await src.uploaded;          // the server's copy of the video
    start = src.in;
    length = src.out - src.in;
  }
  const frames = Math.max(1, Math.round(length * fps));
  const step = Math.max(1, Math.round(S.step));
  // transparent video has to go out as raw RGBA: WebCodecs H.264 has no alpha
  const h264 = !cfg.raw && !cfg.alpha && canvas ? await h264Config(w, h, fps, cfg.crf) : null;

  const enc = await post("/api/encode/open", {
    width: w, height: h, fps, crf: cfg.crf, name: cfg.name, stream: h264 ? "h264" : null, alpha: cfg.alpha || null,
    audio: isVid && cfg.sound ? { id: info.id, start, duration: frames / fps } : null,
  });
  let dec = null, reader = null, sink = null;
  const t0 = performance.now();
  try {
    // The cube itself, sized for this source and the Detail setting.
    const [vw, vh] = app.volumeSize();
    renderer.setVolume(vw, vh, Math.round(S.history));
    if (isVid) {
      dec = await post("/api/decode/open", { id: info.id, fps, width: vw, height: vh, start, duration: frames / fps });
      reader = new FrameReader(dec.sid, dec.width, dec.height);
    }
    if (h264) {
      canvas.width = w;               // the window's canvas renders at full size while exporting
      canvas.height = h;
      sink = new H264Sink(enc.sid, h264, canvas, fps);
    } else {
      sink = new RawSink(enc.sid, renderer, w, h);
    }
    let have = false;
    for (let i = 0; i < frames; i++) {
      if (signal?.aborted) throw abortError();
      const t = i / fps;
      const newLayer = i === 0 || i % step === 0;
      if (isVid) {
        const f = await reader.read();       // every decoded frame is read, so the clock stays exact
        if (f && newLayer) {
          renderer.setSource({ data: f, width: reader.w, height: reader.h });
          const d = app.depthAt?.(start + t) || null;   // this frame's AI depth, when the video has it
          renderer.setDepth(d, d ? d.index : null);
          renderer.setPose(app.poseAt?.(start + t) || null, start + t);
          const k = app.trackAt?.(start + t) || null;      // the followed object, when there is one
          renderer.setMask(k ? k.mask : null, k ? k.key : null, k ? k.point : null);
          have = true;
        }
      } else if (newLayer) {
        const img = src.kind === "photos" ? src.imageAt(t, fps, S.photoHold) : src.imageAt(t);
        renderer.setSource(img, img.width, img.height);
        renderer.setDepth(null);
        renderer.setMask(null);
        renderer.setPose(app.poseAt?.(t) || null, t);
        have = true;
      }
      if (i === 0) {
        if (!have) throw new Error("the video gave no frames");
        renderer.fill();
      } else if (newLayer) {
        renderer.push();
      }
      const st = app.state(t, i);
      st.output = cfg.alpha ? "alpha" : "bg";
      renderer.render(st, w, h, !!h264);
      await sink.add(i);
      onProgress?.(i + 1, frames);
    }
    if (signal?.aborted) throw abortError();
    const res = await sink.finish(frames);
    const secs = (performance.now() - t0) / 1000;
    return { ...res, seconds: frames / fps, width: w, height: h, fps,
      via: h264 ? `WebCodecs ${h264.codec}` : `raw + ${enc.encoder}`, alpha: cfg.alpha || null, renderFps: Math.round((frames / secs) * 10) / 10 };
  } catch (e) {
    await sink?.settle();     // let a frame on its way land before the session goes
    sink?.close();
    await post(`/api/encode/${enc.sid}/close`).catch(() => {});
    throw e;
  } finally {
    await reader?.settle();
    if (dec) await post(`/api/decode/${dec.sid}/close`).catch(() => {});
  }
}
