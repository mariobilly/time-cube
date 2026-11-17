"""AI depth for Time Cube: every frame of a video gets a Depth Anything depth map, made on the GPU.

The server runs this as its own process, one video per run, so the model and its graphics memory are gone again
when it finishes (other apps, like ComfyUI, need that memory):

    python depth.py --src <video> --out <path without extension> [--fps 30] [--long 320]

It writes <out>.bin, every frame's depth one after the other (height x width bytes each, 255 = nearest, 0 = farthest,
so the sky is 0), then measures the camera path (path.py, <out>.path-fov65.json), and last <out>.json with the size
and frame rate: when the .json is there, both are done. Progress goes to stdout as JSON lines:
    {"t": "progress", "done": 40, "total": 300, "ms": 21.5}
    {"t": "done", "frames": 300, ...}       or       {"t": "error", "error": "..."}
Frame k is the picture at k / fps seconds, the same timing the server's export decoder uses.

The model (Depth Anything large, about 1.3 GB) loads from the Hugging Face cache; the first time, it is downloaded
there once. With --parent <pid> (the server passes its own), this process stops at once when that process is gone.
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)
from server import DEPTH_VERSION as VERSION, NO_WINDOW, even, find_tool, no_throttle, probe_media  # noqa: E402

DEPTH_REPO = "LiheYoung/depth-anything-large-hf"
MODEL_LONG = 518      # the model's input: long side in pixels (a multiple of 14, the model's patch size)
BATCH = 8
PATH_FOV = 65         # the lens assumed for the camera path (the window can ask for another)
LOW, HIGH = 0.02, 0.98  # each frame's depth is stretched between these quantiles (robust to specks and the sky)
SMOOTH_S = 2.0          # ...which are smoothed over this many seconds either side


def say(**msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def sizes(w, h, long_side):
    """(model input w, h) and (stored depth w, h) for a w x h video."""
    a = w / max(1, h)
    def m14(v):
        return max(14 * 8, int(round(v / 14.0)) * 14)
    if a >= 1:
        model = (m14(MODEL_LONG), m14(MODEL_LONG / a))
        store = (even(long_side), even(long_side / a))
    else:
        model = (m14(MODEL_LONG * a), m14(MODEL_LONG))
        store = (even(long_side * a), even(long_side))
    return model, store


def watch_parent(pid):
    """Stop when the server that started us is gone.

    It waits on the server's process handle. (Not on stdin: on Windows a thread blocked reading a pipe makes every
    new child process, like ffmpeg, hang until the read ends.)
    """
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes as wt
    k = ctypes.windll.kernel32
    k.OpenProcess.restype = wt.HANDLE
    k.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
    k.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]
    h = k.OpenProcess(0x00100000, False, int(pid))   # SYNCHRONIZE
    if not h:
        os._exit(3)                                     # already gone
    def run():
        k.WaitForSingleObject(h, 0xFFFFFFFF)
        os._exit(3)
    threading.Thread(target=run, daemon=True).start()


def load_model():
    import torch
    from transformers import AutoModelForDepthEstimation
    if not torch.cuda.is_available():
        raise RuntimeError("AI depth needs the NVIDIA graphics card, and PyTorch can't see it")
    try:
        m = AutoModelForDepthEstimation.from_pretrained(DEPTH_REPO, local_files_only=True, dtype=torch.float16)
    except OSError:     # not in the cache yet: download it once
        try:
            m = AutoModelForDepthEstimation.from_pretrained(DEPTH_REPO, dtype=torch.float16)
        except OSError as e:
            raise RuntimeError(f"the Depth Anything model ({DEPTH_REPO}) couldn't be downloaded: {e}")
    return m.to("cuda").eval()


def frames_from(ffmpeg, src, fps, w, h, out_q):
    """Decode the video at `fps`, scaled to w x h RGB, into out_q as batches (None at the end)."""
    import numpy as np
    cmd = [ffmpeg, "-v", "error", "-nostdin", "-i", src, "-an", "-sn", "-vf",
           f"fps={fps},scale={w}:{h}:flags=bicubic,format=rgb24", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                            creationflags=NO_WINDOW)
    no_throttle(proc.pid)
    size = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(size * BATCH)
            n = len(buf) // size
            if n:
                out_q.put(np.frombuffer(bytearray(buf[:n * size]), np.uint8).reshape(n, h, w, 3))
            if n < BATCH:
                break
        proc.wait()
        if proc.returncode not in (0, None):
            err = proc.stderr.read().decode("utf-8", "replace").strip()[-300:]
            out_q.put(RuntimeError(f"ffmpeg couldn't read the video: {err or proc.returncode}"))
    except Exception as e:  # noqa: BLE001
        out_q.put(e)
    finally:
        try:
            proc.kill()
        except Exception:
            pass
        out_q.put(None)


def smooth(values, k):
    """A centred moving average over 2k+1 frames (the ends hold their value)."""
    import numpy as np
    v = np.asarray(values, np.float64)
    if k <= 0 or len(v) < 2:
        return v
    p = np.pad(v, (k, k), mode="edge")
    c = np.cumsum(np.insert(p, 0, 0.0))
    return (c[2 * k + 1:] - c[:-2 * k - 1]) / (2 * k + 1)


def run(src, out, fps, long_side):
    import numpy as np
    import torch
    import torch.nn.functional as F

    ffmpeg, ffprobe = find_tool("ffmpeg"), find_tool("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg was not found on this computer")
    info = probe_media(ffprobe, src)
    (mw, mh), (sw, sh) = sizes(info["width"], info["height"], long_side)
    total = max(1, int(round(info["duration"] * fps)))
    say(t="progress", done=0, total=total, ms=None, stage="loading the depth model")
    t0 = time.time()
    model = load_model()
    load_s = time.time() - t0

    q = queue.Queue(maxsize=4)
    threading.Thread(target=frames_from, args=(ffmpeg, src, fps, mw, mh, q), daemon=True).start()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda", dtype=torch.float16).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda", dtype=torch.float16).view(1, 3, 1, 1)
    quant = torch.tensor([LOW, HIGH], device="cuda")
    raw_path = out + ".raw.part"     # pass 1: the model's own depth, as float16
    done, gpu_s, lows, highs = 0, 0.0, [], []
    try:
        with open(raw_path, "wb") as fh, torch.inference_mode():
            while True:
                batch = q.get()
                if batch is None:
                    break
                if isinstance(batch, Exception):
                    raise batch
                t1 = time.time()
                x = torch.from_numpy(batch).to("cuda", non_blocking=True).permute(0, 3, 1, 2).half().div_(255)
                x = (x - mean) / std
                d = model(pixel_values=x).predicted_depth            # (n, mh, mw): bigger = nearer
                d = F.interpolate(d.unsqueeze(1).float(), size=(sh, sw), mode="area").squeeze(1)
                lohi = torch.quantile(d.reshape(d.shape[0], -1), quant, dim=1).cpu().numpy()   # (2, n)
                lows += lohi[0].tolist()
                highs += lohi[1].tolist()
                fh.write(d.half().cpu().numpy().tobytes())
                gpu_s += time.time() - t1
                done += batch.shape[0]
                say(t="progress", done=done, total=max(total, done), ms=round(1000 * gpu_s / done, 1))
        if done == 0:
            raise RuntimeError("the video gave no frames")
        del model
        # pass 2: each frame stretched between its near and far, both smoothed over SMOOTH_S seconds, so the scene
        # keeps its depth when someone walks up to the camera (the model's own scale is already steady)
        lo, hi = smooth(lows, int(round(SMOOTH_S * fps))), smooth(highs, int(round(SMOOTH_S * fps)))
        raw = np.memmap(raw_path, np.float16, "r", shape=(done, sh, sw))
        with open(out + ".bin.part", "wb") as fh:
            for i in range(0, done, 64):
                r = raw[i:i + 64].astype(np.float32)
                a, b = lo[i:i + 64, None, None], hi[i:i + 64, None, None]
                fh.write(np.clip(np.rint((r - a) / np.maximum(b - a, 1e-6) * 255), 0, 255).astype(np.uint8).tobytes())
        del raw
    finally:
        try:
            os.remove(raw_path)
        except OSError:
            pass
    meta = {"version": VERSION, "model": DEPTH_REPO, "fps": fps, "frames": done, "width": sw, "height": sh,
            "source": {"width": info["width"], "height": info["height"], "duration": info["duration"]},
            "msPerFrame": round(1000 * gpu_s / done, 1), "seconds": round(time.time() - t0, 1),
            "modelLoadSeconds": round(load_s, 1)}
    os.replace(out + ".bin.part", out + ".bin")
    # the camera path, with the usual lens; a failure here doesn't cost the depth (the path can be measured later)
    say(t="progress", done=done, total=done, ms=meta["msPerFrame"], stage="measuring the camera path")
    try:
        import path
        p = path.run(src, out, PATH_FOV, meta)
        meta["path"] = {"steps": p["steps"], "solved": p["solved"], "seconds": p["seconds"]}
    except Exception as e:  # noqa: BLE001
        meta["path"] = {"error": str(e) or e.__class__.__name__}
    with open(out + ".json.part", "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    os.replace(out + ".json.part", out + ".json")   # the .json appears last: its presence means "complete"
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--long", type=int, default=320)
    ap.add_argument("--parent", type=int, help="stop when this process (the server) is gone")
    a = ap.parse_args()
    if a.parent:
        watch_parent(a.parent)
    no_throttle()
    try:
        meta = run(a.src, a.out, a.fps, a.long)
    except Exception as e:  # noqa: BLE001 - every failure goes back to the window as text
        for leftover in (a.out + ".bin.part", a.out + ".json.part"):
            try:
                os.remove(leftover)
            except OSError:
                pass
        say(t="error", error=str(e) or e.__class__.__name__)
        sys.exit(1)
    say(t="done", **meta)


if __name__ == "__main__":
    main()
