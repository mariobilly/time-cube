"""Object tracking for Time Cube: find a thing in a frame, cut it out, and follow it through the whole video.

The server keeps this running while the Track section is used (and closes it after a few idle minutes, so its
graphics memory is freed):

    python track.py --serve [--parent <server pid>]

It reads one JSON command per line on stdin and answers with JSON lines on stdout (a follow also reports progress):
    {"cmd": "find", "src", "fps", "frame", "text"}     -> {"t": "result", "candidates": [...]}
    {"cmd": "here", "src", "fps", "frame"}             -> {"t": "result", "labels": [{"label", "score"}]}
    {"cmd": "cut",  "src", "fps", "frame", "point"}    -> {"t": "result", "candidates": [one]}
    {"cmd": "follow", "src", "fps", "frames", "frame", "box", "point", "outline", "label", "out", "size", "depth"}
                                                       -> {"t": "progress", ...} ..., {"t": "result", ...}
Frames are numbered at `fps` from the start of the video, the same grid as its AI depth. A candidate is
{"box": [x0, y0, x1, y1], "outline": [[x, y], ...], "score", "area"} in 0..1 picture coordinates.

Finding by text: CLIPSeg (CIDAS/clipseg-rd64-refined) marks where the words fit; the strongest places are cut out
precisely with SAM (facebook/sam-vit-base). Following starts from exactly the outline that was picked. The points
inside the object are followed to the next frame (optical flow), which moves its box; SAM cuts the object out again
inside that box, and of SAM's three cut-outs (a part ... the whole) the one most like the last frame's is kept, so
it doesn't jump from the shirt to the whole person. If none is like it, SAM is unsure, or the object shrinks to a
sliver of its size in the last second, it counts as lost from there on (it left the shot or something covered it).
A follow writes <out>.bin (a mask per frame, `size` = [w, h] bytes each, 255 = the object) and then <out>.json.
Both models load from the Hugging Face cache; the first time, they are downloaded there once (about 1 GB).
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)
from server import NO_WINDOW, find_tool, no_throttle  # noqa: E402

TRACK_VERSION = 1
FRAME_LONG = 640        # the frames objects are found and followed in: long side in pixels
CHUNK = 96              # frames decoded at a time while following
SEG_REPO = "CIDAS/clipseg-rd64-refined"
SAM_REPO = "facebook/sam-vit-base"
SAM_MEAN = (123.675, 116.28, 103.53)
SAM_STD = (58.395, 57.12, 57.375)
# "What's in this moment": everyday things CLIPSeg is asked about
VOCAB = ["person", "man", "woman", "child", "face", "hand", "hat", "bag", "phone", "car", "bicycle", "motorbike",
         "bus", "boat", "dog", "cat", "bird", "horse", "tree", "palm tree", "flower", "building", "window", "door",
         "road", "sky", "sea", "water", "mountain", "bench", "chair", "table", "lamp", "railing", "fence", "ball",
         "bottle", "cup", "book", "screen", "guitar", "shoe"]


def say(**msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def frame_size(w, h):
    a = w / max(1, h)
    return (FRAME_LONG, int(round(FRAME_LONG / a / 2)) * 2) if a >= 1 else (int(round(FRAME_LONG * a / 2)) * 2, FRAME_LONG)


def probe_size(src):
    from server import probe_media
    info = probe_media(find_tool("ffprobe"), src)
    return info["width"], info["height"]


def read_frames(src, fps, first, count, w, h):
    """`count` frames from frame `first` on (fewer at the end), RGB (n, h, w, 3)."""
    cmd = [find_tool("ffmpeg"), "-v", "error", "-nostdin"]
    if first > 0:
        cmd += ["-ss", f"{first / fps:.4f}"]
    cmd += ["-i", src, "-an", "-sn", "-frames:v", str(count), "-vf", f"fps={fps},scale={w}:{h}:flags=area,format=rgb24",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg couldn't read the video: " + r.stderr.decode("utf-8", "replace")[-300:])
    n = len(r.stdout) // (w * h * 3)
    return np.frombuffer(r.stdout[:n * w * h * 3], np.uint8).reshape(n, h, w, 3)


class Models:
    """CLIPSeg and SAM, loaded on first use and kept on the GPU."""

    def __init__(self):
        self.seg = self.sam = None

    def torch(self):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("tracking needs the NVIDIA graphics card, and PyTorch can't see it")
        return torch

    def load_seg(self):
        if self.seg is None:
            torch = self.torch()
            from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
            try:
                p = CLIPSegProcessor.from_pretrained(SEG_REPO, local_files_only=True)
                m = CLIPSegForImageSegmentation.from_pretrained(SEG_REPO, local_files_only=True, dtype=torch.float16)
            except OSError:     # not in the cache yet: download it once
                try:
                    p = CLIPSegProcessor.from_pretrained(SEG_REPO)
                    m = CLIPSegForImageSegmentation.from_pretrained(SEG_REPO, dtype=torch.float16)
                except OSError as e:
                    raise RuntimeError(f"the CLIPSeg model ({SEG_REPO}) couldn't be downloaded: {e}")
            self.seg = (p, m.to("cuda").eval())
        return self.seg

    def load_sam(self):
        if self.sam is None:
            torch = self.torch()
            from transformers import SamModel
            try:
                m = SamModel.from_pretrained(SAM_REPO, local_files_only=True, dtype=torch.float16)
            except OSError:     # not in the cache yet: download it once
                try:
                    m = SamModel.from_pretrained(SAM_REPO, dtype=torch.float16)
                except OSError as e:
                    raise RuntimeError(f"the SAM model ({SAM_REPO}) couldn't be downloaded: {e}")
            self.sam = m.to("cuda").eval()
        return self.sam

    # ---- CLIPSeg ------------------------------------------------------------------------------------------
    def heat(self, img, prompts):
        """How well each prompt fits each place: (n, h, w) 0..1 at the frame's size."""
        import cv2
        torch = self.torch()
        from PIL import Image
        p, m = self.load_seg()
        with torch.inference_mode():
            inp = p(text=prompts, images=[Image.fromarray(img)] * len(prompts), padding=True, return_tensors="pt")
            inp = {k: v.to("cuda") for k, v in inp.items()}
            inp["pixel_values"] = inp["pixel_values"].half()
            lg = m(**inp).logits
            lg = lg[None] if lg.dim() == 2 else lg
            prob = torch.sigmoid(lg.float()).cpu().numpy()
        return np.stack([cv2.resize(x, (img.shape[1], img.shape[0])) for x in prob])

    # ---- SAM ----------------------------------------------------------------------------------------------
    def embed(self, img):
        """SAM's view of a frame (done once per frame, on the GPU): (embedding, scale, resized size)."""
        torch = self.torch()
        import torch.nn.functional as F
        sam = self.load_sam()
        h, w = img.shape[:2]
        s = 1024.0 / max(h, w)
        nh, nw = int(round(h * s)), int(round(w * s))
        with torch.inference_mode():
            x = torch.from_numpy(np.array(img)).to("cuda").permute(2, 0, 1)[None].float()   # a writable copy
            x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)
            mean = torch.tensor(SAM_MEAN, device="cuda").view(1, 3, 1, 1)
            std = torch.tensor(SAM_STD, device="cuda").view(1, 3, 1, 1)
            x = F.pad((x - mean) / std, (0, 1024 - nw, 0, 1024 - nh))
            emb = sam.get_image_embeddings(x.half())
        return emb, s, (nh, nw)

    def segment(self, view, size, box=None, point=None, like=None):
        """One of SAM's three cut-outs for a box and/or a point (frame pixels): (mask bool (h, w), score, overlap).
        Without `like`, the one SAM is surest of; with it (a mask), the one that overlaps it most."""
        torch = self.torch()
        import torch.nn.functional as F
        sam = self.load_sam()
        emb, s, (nh, nw) = view
        h, w = size
        kw = {}
        if box is not None:
            kw["input_boxes"] = torch.tensor([[[float(v) * s for v in box]]], device="cuda", dtype=torch.float16)
        if point is not None:
            kw["input_points"] = torch.tensor([[[[point[0] * s, point[1] * s]]]], device="cuda", dtype=torch.float16)
            kw["input_labels"] = torch.ones((1, 1, 1), device="cuda", dtype=torch.long)
        with torch.inference_mode():
            o = sam(image_embeddings=emb, multimask_output=True, **kw)
            scores = o.iou_scores[0, 0].float().cpu().numpy()
            m = o.pred_masks[0, 0][:, None].float()                                    # (3, 1, 256, 256)
            m = F.interpolate(m, size=(1024, 1024), mode="bilinear", align_corners=False)[..., :nh, :nw]
            m = (F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)[:, 0] > 0).cpu().numpy()
        if like is None:
            best = int(scores.argmax())
            return m[best], float(scores[best]), 1.0
        inter = (m & like[None]).sum((1, 2))
        union = (m | like[None]).sum((1, 2))
        overlap = inter / np.maximum(union, 1)
        best = int(np.argmax(overlap + 0.05 * scores))
        return m[best], float(scores[best]), float(overlap[best])


M = Models()


def bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 12:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float64)


def candidate(mask, score):
    """A cut-out as the window shows it: box, outline (the biggest contours), area, all in 0..1."""
    import cv2
    h, w = mask.shape
    b = bbox(mask)
    if b is None:
        return None
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = sorted(cs, key=cv2.contourArea, reverse=True)[:4]
    outline = []
    for c in cs:
        if cv2.contourArea(c) < 0.02 * cv2.contourArea(cs[0]):
            break
        c = cv2.approxPolyDP(c, 1.2, True)[:, 0]
        outline.append([[round(float(x) / w, 4), round(float(y) / h, 4)] for x, y in c])
    return {"box": [round(b[0] / w, 4), round(b[1] / h, 4), round(b[2] / w, 4), round(b[3] / h, 4)],
            "outline": outline, "score": round(score, 3), "area": round(float(mask.mean()), 4)}


def one_frame(src, fps, frame):
    w, h = frame_size(*probe_size(src))
    f = read_frames(src, fps, frame, 1, w, h)
    if not len(f):
        raise RuntimeError("that moment of the video can't be read")
    return f[0]


def find(src, fps, frame, text, most=3):
    """Where `text` is in the frame: up to `most` cut-outs, the best first."""
    import cv2
    img = one_frame(src, fps, frame)
    heat = M.heat(img, [text.strip()[:120]])[0]
    peak = float(heat.max())
    if peak < 0.2:
        return []
    n, lab, stats, _ = cv2.connectedComponentsWithStats((heat > max(0.3, peak * 0.5)).astype(np.uint8))
    places = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 30:
            continue
        blob = lab == i
        yx = np.unravel_index(np.argmax(np.where(blob, heat, 0)), heat.shape)
        places.append((float(heat[blob].max()) * np.sqrt(area), (x, y, x + bw, y + bh), (yx[1], yx[0]), blob))
    places.sort(key=lambda p: -p[0])
    view = M.embed(img)
    out = []
    for score, box, top, blob in places[:most]:
        # SAM is given the place and its surest spot; of its three cut-outs (a part ... the whole), the one most
        # like the place the words were found in is kept
        mask, conf, _ = M.segment(view, img.shape[:2], box=box, point=top, like=blob)
        c = candidate(mask, conf * min(1.0, peak / 0.6))
        if c:
            c["label"] = text.strip()[:60]
            out.append(c)
    return out


def here(src, fps, frame):
    """The everyday things CLIPSeg sees in the frame, the clearest first."""
    img = one_frame(src, fps, frame)
    heat = M.heat(img, VOCAB)
    found = []
    for word, x in zip(VOCAB, heat):
        top, area = float(np.percentile(x, 99.5)), float((x > 0.4).mean())
        if top > 0.5 and area > 0.002:
            found.append({"label": word, "score": round(top, 3)})
    found.sort(key=lambda f: -f["score"])
    # "man" and "person" for the same figure: keep the more specific word first, drop near-duplicates
    return found[:12]


def cut(src, fps, frame, point):
    img = one_frame(src, fps, frame)
    h, w = img.shape[:2]
    mask, conf, _ = M.segment(M.embed(img), (h, w), point=(point[0] * w, point[1] * h))
    c = candidate(mask, conf)
    if c:
        c["label"] = "your pick"
    return [c] if c else []


def follow(src, fps, frames, frame, box, point, out, size, depth_bin=None, label="", outline=None):
    """Follow the object from `frame` to both ends of the video. Writes the masks and the per-frame info."""
    import cv2
    t0 = time.time()
    w, h = frame_size(*probe_size(src))
    mw, mh = size
    masks = np.zeros((frames, mh, mw), np.uint8)
    info = [None] * frames
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    near = None
    if depth_bin and os.path.exists(depth_bin):
        near = np.memmap(depth_bin, np.uint8, "r", shape=(frames, mh, mw))

    def keep(k, mask, score):
        small = cv2.resize(mask.astype(np.float32), (mw, mh), interpolation=cv2.INTER_AREA)
        masks[k] = np.clip(small * 255 + 0.5, 0, 255).astype(np.uint8)
        ys, xs = np.nonzero(mask)
        b = bbox(mask)
        nv = None
        if near is not None:
            sel = masks[k] > 127
            if sel.any():
                nv = round(float(np.median(near[k][sel])) / 255, 4)
        info[k] = {"cx": round(float(xs.mean()) / w, 4), "cy": round(float(ys.mean()) / h, 4),
                   "box": [round(b[0] / w, 4), round(b[1] / h, 4), round(b[2] / w, 4), round(b[3] / h, 4)],
                   "area": round(float(mask.mean()), 5), "near": nv, "score": round(score, 3)}

    done = [0]

    def tick():
        done[0] += 1
        if done[0] % 4 == 0 or done[0] == frames:
            say(t="progress", done=done[0], total=frames, seconds=round(time.time() - t0, 1))

    # the seed frame: exactly the outline that was picked (else SAM from the box / point)
    img = read_frames(src, fps, frame, 1, w, h)[0]
    if outline:
        seed = np.zeros((h, w), np.uint8)
        polys = [np.round(np.array(poly, np.float64) * [w, h]).astype(np.int32) for poly in outline if len(poly) >= 3]
        cv2.fillPoly(seed, polys, 1)
        seed, score = seed.astype(bool), 1.0
    else:
        bx = None if box is None else [box[0] * w, box[1] * h, box[2] * w, box[3] * h]
        pt = None if point is None else (point[0] * w, point[1] * h)
        seed, score, _ = M.segment(M.embed(img), (h, w), box=bx, point=pt)
    if bbox(seed) is None:
        raise RuntimeError("nothing to follow there")
    keep(frame, seed, score)
    tick()
    gray0 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def run(direction):
        prev, gray_prev = seed, gray0
        k = frame
        lost = False
        sizes = [int(seed.sum())]   # the object's size in the last second, to notice when it collapses
        while not lost:
            first = k + 1 if direction > 0 else max(0, k - CHUNK)
            count = min(CHUNK, frames - first) if direction > 0 else k - first
            if count <= 0:
                return
            chunk = read_frames(src, fps, first, count, w, h)
            order = range(len(chunk)) if direction > 0 else range(len(chunk) - 1, -1, -1)
            for j in order:
                nk = first + j
                cur = chunk[j]
                gray = cv2.cvtColor(cur, cv2.COLOR_RGB2GRAY)
                b = bbox(prev)
                pts = cv2.goodFeaturesToTrack(gray_prev, 300, 0.01, 4, mask=prev.astype(np.uint8) * 255)
                shift = np.zeros(2)
                if pts is not None and len(pts) > 5:
                    p1, st, _ = cv2.calcOpticalFlowPyrLK(gray_prev, gray, pts, None, **lk)
                    ok = st[:, 0] == 1
                    if ok.sum() > 5:
                        shift = np.median((p1 - pts)[ok, 0], axis=0)
                pad = 0.06 * np.array([b[2] - b[0], b[3] - b[1]] * 2) * np.array([-1, -1, 1, 1])
                nb = np.clip(b + np.r_[shift, shift] + pad, 0, [w, h, w, h])
                cy, cx = np.argwhere(prev).mean(0)
                moved = cv2.warpAffine(prev.astype(np.uint8), np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]),
                                       (w, h), flags=cv2.INTER_NEAREST).astype(bool)
                m, sc, overlap = M.segment(M.embed(cur), (h, w), box=nb, point=(cx + shift[0], cy + shift[1]),
                                           like=moved)
                ratio = m.sum() / max(prev.sum(), 1)
                shrunk = m.sum() < 0.12 * np.median(sizes[-int(fps):])   # a sliver of what it was: it's leaving
                if sc < 0.5 or overlap < 0.25 or ratio < 0.25 or ratio > 4 or shrunk or bbox(m) is None:
                    lost = True       # it left the shot, or something covered it
                    print(f"lost at frame {nk} going {'forward' if direction > 0 else 'back'}: SAM score {sc:.2f},"
                          f" overlap {overlap:.2f}, size x{ratio:.2f}", file=sys.stderr, flush=True)
                    break
                keep(nk, m, sc)
                tick()
                sizes.append(int(m.sum()))
                prev, gray_prev, k = m, gray, nk
            if direction > 0 and len(chunk) < count:
                return

    run(+1)
    run(-1)
    seen = sum(1 for i in info if i)
    meta = {"version": TRACK_VERSION, "label": label, "fps": fps, "frames": frames, "width": mw, "height": mh,
            "seed": frame, "seen": seen, "seconds": round(time.time() - t0, 1), "info": info}
    masks.tofile(out + ".bin.part")
    os.replace(out + ".bin.part", out + ".bin")
    with open(out + ".json.part", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, separators=(",", ":"))
    os.replace(out + ".json.part", out + ".json")
    return {k: v for k, v in meta.items() if k != "info"}


def serve():
    """One command per stdin line, one answer per command (progress lines before it for a follow)."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            c = json.loads(line)
            cmd = c.get("cmd")
            if cmd == "find":
                say(t="result", candidates=find(c["src"], c["fps"], int(c["frame"]), c["text"]))
            elif cmd == "here":
                say(t="result", labels=here(c["src"], c["fps"], int(c["frame"])))
            elif cmd == "cut":
                say(t="result", candidates=cut(c["src"], c["fps"], int(c["frame"]), c["point"]))
            elif cmd == "follow":
                say(t="result", **follow(c["src"], c["fps"], int(c["frames"]), int(c["frame"]), c.get("box"),
                                         c.get("point"), c["out"], c["size"], c.get("depth"), c.get("label", ""),
                                         c.get("outline")))
            elif cmd == "ping":
                say(t="result", ok=True)
            else:
                raise ValueError(f"unknown command {cmd!r}")
        except Exception as e:  # noqa: BLE001 - the server shows it in the window
            say(t="error", error=str(e) or e.__class__.__name__)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--parent", type=int, help="stop when this process (the server) is gone")
    a = ap.parse_args()
    if a.parent:
        from depth import watch_parent
        watch_parent(a.parent)
    no_throttle()
    if a.serve:
        serve()


if __name__ == "__main__":
    main()
