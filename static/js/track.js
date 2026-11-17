// The Track section's picker: the paused frame over the picture, with what was found outlined. A click on an
// outline picks it; a click anywhere else asks the server to cut out what is under the pointer.
// Candidates come from the server: { box: [x0, y0, x1, y1], outline: [[[x, y], ...], ...], label, score } in 0..1.

// Is (x, y) inside the polygon? (even-odd rule)
function inside(poly, x, y) {
  let ok = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) ok = !ok;
  }
  return ok;
}

export class Picker {
  constructor(root, canvas, hint) {
    this.root = root;
    this.canvas = canvas;
    this.hint = hint;
    this.frame = document.createElement("canvas");   // the paused picture, at the video's own size
    this.cands = [];
    this.sel = -1;
    this.onPick = null;      // (index) => {}
    this.onCut = null;       // (u, v) => {}
    canvas.addEventListener("click", (e) => {
      const r = canvas.getBoundingClientRect();
      const u = (e.clientX - r.left) / r.width, v = (e.clientY - r.top) / r.height;
      const i = this.hit(u, v);
      if (i >= 0) this.onPick?.(i);
      else this.onCut?.(u, v);
    });
    new ResizeObserver(() => { if (this.isOpen) this.layout(); }).observe(root);
  }

  get isOpen() { return this.root.classList.contains("show"); }

  // Show a video's current frame (it should be paused).
  open(video) {
    this.frame.width = video.videoWidth;
    this.frame.height = video.videoHeight;
    this.frame.getContext("2d").drawImage(video, 0, 0);
    this.root.classList.add("show");
    this.layout();
  }

  close() { this.root.classList.remove("show"); }

  layout() {
    const box = this.root.getBoundingClientRect();
    const a = this.frame.width / Math.max(1, this.frame.height);
    const maxW = box.width - 40, maxH = box.height - 90;
    const w = Math.max(40, Math.min(maxW, maxH * a)), h = w / a;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    this.canvas.style.width = `${Math.round(w)}px`;
    this.canvas.style.height = `${Math.round(h)}px`;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.draw();
  }

  set(cands, sel) {
    this.cands = cands || [];
    this.sel = sel;
    this.draw();
  }

  hit(u, v) {
    // the selected one first, then the smallest that contains the point (a hat over the man wearing it)
    const order = this.cands.map((c, i) => [i, (c.box[2] - c.box[0]) * (c.box[3] - c.box[1])]).sort((a, b) => a[1] - b[1]);
    if (this.sel >= 0) order.unshift([this.sel, 0]);
    for (const [i] of order) {
      if ((this.cands[i].outline || []).some((poly) => inside(poly, u, v))) return i;
    }
    return -1;
  }

  draw() {
    const c = this.canvas, ctx = c.getContext("2d"), W = c.width, H = c.height;
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(this.frame, 0, 0, W, H);
    this.cands.forEach((cand, i) => {
      const on = i === this.sel;
      ctx.beginPath();
      for (const poly of cand.outline || []) {
        poly.forEach(([x, y], k) => (k ? ctx.lineTo(x * W, y * H) : ctx.moveTo(x * W, y * H)));
        ctx.closePath();
      }
      ctx.fillStyle = on ? "rgba(255, 170, 70, 0.32)" : "rgba(255, 255, 255, 0.14)";
      ctx.fill("evenodd");
      ctx.lineWidth = (on ? 3 : 1.6) * (W / c.clientWidth || 1);
      ctx.strokeStyle = on ? "#ffaa46" : "rgba(255, 255, 255, 0.85)";
      ctx.stroke();
      // its name at the top of its box
      const [x0, y0] = cand.box;
      const label = `${i + 1}. ${cand.label || "thing"}`;
      ctx.font = `${Math.round(12 * (W / c.clientWidth || 1))}px Segoe UI, sans-serif`;
      const tw = ctx.measureText(label).width + 10 * (W / c.clientWidth || 1);
      const lh = 18 * (W / c.clientWidth || 1);
      const lx = Math.min(W - tw, x0 * W), ly = Math.max(0, y0 * H - lh);
      ctx.fillStyle = on ? "#ffaa46" : "rgba(20, 20, 22, 0.75)";
      ctx.fillRect(lx, ly, tw, lh);
      ctx.fillStyle = on ? "#141416" : "#f1efea";
      ctx.fillText(label, lx + 5 * (W / c.clientWidth || 1), ly + lh * 0.72);
    });
  }
}
