// Where the frames come from: a video file, the webcam, a set of photos, or the built-in demo.
// Every source has kind, name, width, height. Live sources give their current picture with image(); photos and the
// demo can also draw the picture for any time with imageAt(t, fps, hold), which the export uses.

export const VIDEO_EXT = /\.(mp4|mov|m4v|webm|mkv|avi|wmv|mpg|mpeg|3gp|ogv)$/i;
export const IMAGE_EXT = /\.(png|jpe?g|webp|gif|bmp|avif|tiff?)$/i;

export function isVideo(file) {
  return (file.type || "").startsWith("video/") || VIDEO_EXT.test(file.name || "");
}

export function isImage(file) {
  return (file.type || "").startsWith("image/") || IMAGE_EXT.test(file.name || "");
}

function waitFor(el, event, errorText) {
  return new Promise((resolve, reject) => {
    const ok = () => { off(); resolve(); };
    const bad = () => { off(); reject(new Error(errorText)); };
    const off = () => { el.removeEventListener(event, ok); el.removeEventListener("error", bad); };
    el.addEventListener(event, ok);
    el.addEventListener("error", bad);
  });
}

export class VideoSource {
  constructor(file) {
    this.kind = "video";
    this.file = file;
    this.name = file.name.replace(/\.[^.]+$/, "");
    this.el = document.createElement("video");
    this.el.playsInline = true;
    this.el.preload = "auto";
    this.el.loop = false;  // the app loops between In and Out itself
    this.url = URL.createObjectURL(file);
  }

  async open() {
    const ready = waitFor(this.el, "loadeddata", "This video can't be played here. Try an MP4 (H.264) file.");
    this.el.src = this.url;
    await ready;
    this.width = this.el.videoWidth;
    this.height = this.el.videoHeight;
    this.duration = Number.isFinite(this.el.duration) ? this.el.duration : 0;
    this.in = 0;
    this.out = this.duration;
    // The server keeps a copy so an export can read every frame exactly. A dropped connection is retried (a
    // second copy is harmless: copies are cleaned up after a few days).
    this.uploaded = (async () => {
      for (let k = 1; ; k++) {
        let r;
        try {
          r = await fetch(`/api/upload?name=${encodeURIComponent(this.file.name)}`, { method: "POST", body: this.file });
        } catch (e) {
          if (k >= 3) throw new Error(`the local server didn't take the video (${e.message})`);
          await new Promise((ok) => setTimeout(ok, 400 * k));
          continue;
        }
        const j = await r.json().catch(() => ({}));
        if (!r.ok || j.error) throw new Error(j.error || "upload failed");
        this.info = j;
        return j;
      }
    })();
    this.uploaded.catch(() => {});
    return this;
  }

  image() { return this.el; }

  stop() {
    this.el.pause();
    this.el.removeAttribute("src");
    this.el.load();
    URL.revokeObjectURL(this.url);
  }
}

export class CameraSource {
  constructor() {
    this.kind = "camera";
    this.name = "webcam";
    this.el = document.createElement("video");
    this.el.playsInline = true;
    this.el.muted = true;
  }

  async open() {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("No camera access in this window");
    this.stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    this.el.srcObject = this.stream;
    await this.el.play();
    if (!this.el.videoWidth) await waitFor(this.el, "loadeddata", "The camera gave no picture");
    this.width = this.el.videoWidth;
    this.height = this.el.videoHeight;
    return this;
  }

  image() { return this.el; }

  stop() {
    for (const t of this.stream?.getTracks() || []) t.stop();
    this.el.srcObject = null;
  }
}

export class PhotoSource {
  constructor(files) {
    this.kind = "photos";
    this.files = [...files].sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
    this.name = this.files.length === 1 ? this.files[0].name.replace(/\.[^.]+$/, "") : "photos";
  }

  async open() {
    this.images = [];
    for (const f of this.files) {
      try {
        this.images.push(await createImageBitmap(f, { imageOrientation: "from-image" }));
      } catch {
        // skip a file that is not a readable picture
      }
    }
    if (!this.images.length) throw new Error("None of these files could be read as a picture");
    this.width = this.images[0].width;
    this.height = this.images[0].height;
    return this;
  }

  // The sequence length in seconds: every photo held for `hold` frames.
  length(fps, hold) { return (this.images.length * hold) / fps; }

  index(t, fps, hold) { return Math.floor((t * fps + 1e-6) / Math.max(1, hold)) % this.images.length; }

  imageAt(t, fps, hold) { return this.images[this.index(t, fps, hold)]; }

  stop() { for (const b of this.images || []) b.close?.(); }
}

// A small animated scene so the cube has something to show before you open anything: a warm room, a dark round
// head bobbing, a pink shirt, a door swinging past. Drawn from the time alone, so it exports frame-exactly.
export class DemoSource {
  constructor() {
    this.kind = "demo";
    this.name = "demo";
    this.width = 640;
    this.height = 480;
    this.canvas = document.createElement("canvas");
    this.canvas.width = this.width;
    this.canvas.height = this.height;
    this.ctx = this.canvas.getContext("2d");
  }

  async open() { this.imageAt(0); return this; }

  imageAt(t) {
    const c = this.ctx, W = this.width, H = this.height;
    const g = c.createLinearGradient(0, 0, W, H);
    g.addColorStop(0, "#f6ead6");
    g.addColorStop(1, "#e2c9a8");
    c.fillStyle = g;
    c.fillRect(0, 0, W, H);
    // door / dark furniture sliding on the right
    const dx = W * (0.7 + 0.12 * Math.sin(t * 0.6));
    c.fillStyle = "#3a2924";
    c.fillRect(dx, H * 0.05, W * 0.2, H * 0.95);
    c.fillStyle = "#c9a36b";
    c.fillRect(dx + W * 0.03, H * 0.48, W * 0.02, H * 0.06);
    // pink shirt
    const hx = W * (0.38 + 0.13 * Math.sin(t * 1.1)), hy = H * (0.42 + 0.05 * Math.sin(t * 2.3));
    c.fillStyle = "#d8a0b5";
    c.beginPath();
    c.ellipse(hx, hy + H * 0.36, W * 0.2, H * 0.24, 0.1 * Math.sin(t), 0, Math.PI * 2);
    c.fill();
    c.fillStyle = "#b9677f";
    for (let i = 0; i < 5; i++) {
      c.fillRect(hx - W * 0.17 + i * W * 0.075, hy + H * 0.2, W * 0.018, H * 0.3);
    }
    // face and hair
    c.fillStyle = "#c58f6b";
    c.beginPath();
    c.ellipse(hx, hy + H * 0.04, W * 0.075, H * 0.11, 0, 0, Math.PI * 2);
    c.fill();
    const hair = c.createRadialGradient(hx, hy - H * 0.04, 4, hx, hy - H * 0.04, W * 0.17);
    hair.addColorStop(0, "#120c0a");
    hair.addColorStop(0.75, "#1d1411");
    hair.addColorStop(1, "rgba(29,20,17,0)");
    c.fillStyle = hair;
    c.beginPath();
    c.ellipse(hx, hy - H * 0.05, W * 0.17, H * 0.15, 0.2 * Math.sin(t * 0.8), 0, Math.PI * 2);
    c.fill();
    // a lamp glow that drifts
    const lx = W * (0.15 + 0.08 * Math.cos(t * 0.45)), ly = H * 0.2;
    const lamp = c.createRadialGradient(lx, ly, 2, lx, ly, W * 0.22);
    lamp.addColorStop(0, "rgba(255,252,240,0.95)");
    lamp.addColorStop(1, "rgba(255,252,240,0)");
    c.fillStyle = lamp;
    c.fillRect(0, 0, W, H);
    return this.canvas;
  }

  stop() {}
}
