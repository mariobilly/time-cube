"""Object tracking tests: a rendered clip whose object is known in every frame.

    python tests/test_track.py        (needs the NVIDIA card and the SAM + CLIPSeg models; skipped without them)

A red ball rolls across a textured wall, past a blue square, and leaves the picture at the right before the end.
Checked: finding it by text, cutting it out with a click, following it (how well the masks cover the true ball,
how far their centres are from it), noticing when it has left, and the same through the server.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.parse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
TMP = tempfile.mkdtemp(prefix="timecube-track-test-")
os.environ["TIME_CUBE_DATA"] = os.path.join(TMP, "data")
os.environ["TIME_CUBE_OUT"] = os.path.join(TMP, "out")
sys.path.insert(0, APP)
sys.path.insert(0, HERE)
import server  # noqa: E402
import test_server  # noqa: E402
from test_server import call  # noqa: E402

PORT = 47976
test_server.PORT = PORT
W, H, FPS, N = 640, 360, 24, 96
R = 34                         # the ball's radius in pixels
LEAVES = 84                    # from this frame on, the ball is out of the picture


def ball_at(i):
    """The ball's centre in frame i: across the wall on a gentle arc, then off the right edge."""
    s = i / (LEAVES - 1)
    return 70 + s * (W + R - 70), 200 - 70 * np.sin(np.pi * min(s, 1.0))


def render(i):
    rng = np.random.default_rng(3)
    blocks = rng.integers(70, 190, (H // 24 + 1, W // 24 + 1, 3)).astype(np.float32)
    wall = np.kron(blocks, np.ones((24, 24, 1)))[:H, :W] * 0.8
    wall += np.random.default_rng(100 + i).normal(0, 3, wall.shape)
    yy, xx = np.mgrid[0:H, 0:W]
    img = wall.copy()
    # a blue square that stays put (not a ball)
    sq = (np.abs(xx - 470) < 36) & (np.abs(yy - 250) < 36)
    img[sq] = (40, 70, 200)
    cx, cy = ball_at(i)
    d = np.hypot(xx - cx, yy - cy)
    on = d < R
    shade = np.clip(1.15 - ((xx - cx + 12) ** 2 + (yy - cy + 12) ** 2) / (2.2 * R * R), 0.35, 1.15)
    img[on] = np.stack([235 * shade, 40 * shade, 35 * shade], -1)[on]
    return np.clip(img, 0, 255).astype(np.uint8), on


def iou(a, b):
    u = (a | b).sum()
    return 1.0 if u == 0 else (a & b).sum() / u


def gpu_ready():
    try:
        import torch
        from huggingface_hub import try_to_load_from_cache
        import track
        return torch.cuda.is_available() and all(
            isinstance(try_to_load_from_cache(r, "config.json"), str) for r in (track.SAM_REPO, track.SEG_REPO))
    except Exception:
        return False


@unittest.skipUnless(gpu_ready(), "no NVIDIA card, or SAM / CLIPSeg isn't on this PC")
class TrackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import track
        cls.T = track
        cls.clip = os.path.join(TMP, "ball.mp4")
        frames, cls.truth = zip(*(render(i) for i in range(N)))
        ff = server.find_tool("ffmpeg")
        p = subprocess.Popen([ff, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r",
                              str(FPS), "-i", "pipe:0", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", cls.clip],
                             stdin=subprocess.PIPE)
        p.stdin.write(np.array(frames).tobytes())
        p.stdin.close()
        assert p.wait() == 0

    def truth_box(self, i):
        cx, cy = ball_at(i)
        return [max(0, cx - R) / W, max(0, cy - R) / H, min(W, cx + R) / W, min(H, cy + R) / H]

    def test_a_find_by_text(self):
        t = time.time()
        c = self.T.find(self.clip, FPS, 30, "red ball")
        print(f"\n  find 'red ball': {len(c)} found, {time.time() - t:.1f} s (with loading the models)", flush=True)
        self.assertTrue(c, "the ball was found")
        b, tb = c[0]["box"], self.truth_box(30)
        inter = max(0, min(b[2], tb[2]) - max(b[0], tb[0])) * max(0, min(b[3], tb[3]) - max(b[1], tb[1]))
        union = (b[2] - b[0]) * (b[3] - b[1]) + (tb[2] - tb[0]) * (tb[3] - tb[1]) - inter
        self.assertGreater(inter / union, 0.7, f"the best one is the ball: {b} vs {tb}")
        self.assertTrue(c[0]["outline"] and len(c[0]["outline"][0]) >= 6)

    def test_b_cut_by_click(self):
        cx, cy = ball_at(40)
        c = self.T.cut(self.clip, FPS, 40, [cx / W, cy / H])
        self.assertEqual(len(c), 1)
        self.assertLess(abs(c[0]["area"] - np.pi * R * R / (W * H)), 0.4 * np.pi * R * R / (W * H), c[0]["area"])

    def test_c_here(self):
        labels = [x["label"] for x in self.T.here(self.clip, FPS, 30)]
        self.assertIn("ball", labels, labels)

    def test_d_follow(self):
        out = os.path.join(TMP, "follow")
        mw, mh = 320, 180
        t = time.time()
        meta = self.T.follow(self.clip, FPS, N, 30, self.truth_box(30), None, out, [mw, mh], None, "red ball")
        took = time.time() - t
        masks = np.fromfile(out + ".bin", np.uint8).reshape(N, mh, mw) > 127
        with open(out + ".json", encoding="utf-8") as fh:
            info = json.load(fh)["info"]
        ious, errs = [], []
        whole = [i for i in range(N) if ball_at(i)[0] + R < W]   # frames with all of the ball in the picture
        for i in whole:
            truth = self.truth[i][::2, ::2]
            ious.append(iou(masks[i], truth))
            if info[i]:
                cx, cy = ball_at(i)
                errs.append(np.hypot(info[i]["cx"] * W - cx, info[i]["cy"] * H - cy))
        print(f"  follow: seen in {meta['seen']} of {N} frames, {took:.1f} s ({1000 * took / N:.0f} ms a frame), "
              f"overlap {np.mean(ious):.3f} (worst {np.min(ious):.3f}), centre off by {np.mean(errs):.1f} px "
              f"(worst {np.max(errs):.1f})", flush=True)
        self.assertGreater(np.mean(ious), 0.85)
        self.assertGreater(np.min(ious), 0.6)
        self.assertLess(np.mean(errs), 3.0)
        # it has left the picture: those frames are empty, not stuck on something else
        self.assertTrue(all(info[i] is None for i in range(LEAVES + 1, N)), [i for i in range(LEAVES, N) if info[i]])
        self.assertFalse(masks[LEAVES + 2:].any())
        self.assertIsNone(info[30]["near"])   # no AI depth here

    def test_e_server(self):
        srv = server.make_server(PORT)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            with open(self.clip, "rb") as fh:
                up = call("/api/upload?name=ball.mp4", fh.read())
            self.assertEqual(call(f"/api/track/status?id={up['id']}")["state"], "none")
            found = call("/api/track/find", {"id": up["id"], "time": 30 / FPS, "text": "red ball"})
            self.assertTrue(found["candidates"], found)
            self.assertIn("error", call("/api/track/find", {"id": up["id"], "time": 1, "text": " "}))
            st = call("/api/track/follow", {"id": up["id"], "time": 30 / FPS, "box": found["candidates"][0]["box"],
                                            "label": "red ball"})
            self.assertEqual(st["state"], "running")
            busy = call("/api/track/here", {"id": up["id"], "time": 1})
            self.assertIn("being followed", busy.get("error", ""))
            end = time.time() + 300
            while time.time() < end and st["state"] == "running":
                time.sleep(0.5)
                st = call(f"/api/track/status?id={up['id']}")
            self.assertEqual(st["state"], "ready", st)
            self.assertEqual(st["frames"], N)
            self.assertGreaterEqual(st["seen"], LEAVES - 2)
            data, headers = call(f"/api/track/data?id={up['id']}", raw=True)
            self.assertEqual(len(data), N * int(headers["X-Width"]) * int(headers["X-Height"]))
            info = call(f"/api/track/info?id={up['id']}")
            self.assertEqual(len(info["info"]), N)
            # the same video opened again has its track
            with open(self.clip, "rb") as fh:
                up2 = call(f"/api/upload?name={urllib.parse.quote('ball again.mp4')}", fh.read())
            self.assertEqual(call(f"/api/track/status?id={up2['id']}")["state"], "ready")
            self.assertEqual(call("/api/track/clear", {"id": up2["id"]})["state"], "none")
            self.assertIn("error", call(f"/api/track/data?id={up['id']}"))
        finally:
            server.TRACKER.kill()
            srv.shutdown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
