"""AI depth tests: the frame-rate plan, a real Depth Anything run through the server (with its camera path), the
cache, stopping a run.

    python tests/test_depth.py        (needs the NVIDIA card and the Depth Anything model; skipped without them)
"""
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
TMP = tempfile.mkdtemp(prefix="timecube-depth-test-")
os.environ["TIME_CUBE_DATA"] = os.path.join(TMP, "data")
os.environ["TIME_CUBE_OUT"] = os.path.join(TMP, "out")
sys.path.insert(0, APP)
sys.path.insert(0, HERE)
import server  # noqa: E402
from test_server import call, make_clip  # noqa: E402
import test_server  # noqa: E402

PORT = 47970
test_server.PORT = PORT   # call() uses it


def gpu_ready():
    try:
        import torch
        from huggingface_hub import try_to_load_from_cache
        import depth
        return torch.cuda.is_available() and isinstance(try_to_load_from_cache(depth.DEPTH_REPO, "config.json"), str)
    except Exception:
        return False


def wait(mid, until=("ready", "error", "cancelled"), timeout=240):
    end = time.time() + timeout
    while time.time() < end:
        st = call(f"/api/depth/status?id={mid}")
        if st.get("state") in until:
            return st
        time.sleep(0.3)
    raise AssertionError(f"depth still {st} after {timeout} s")


class PlanTest(unittest.TestCase):
    def test_frame_rate(self):
        self.assertEqual(server.depth_plan({"fps": 24, "duration": 5}), (24.0, 120))
        self.assertEqual(server.depth_plan({"fps": 60, "duration": 10}), (30.0, 300))
        fps, frames = server.depth_plan({"fps": 30, "duration": 600})   # 10 minutes: fewer depth frames a second
        self.assertLessEqual(frames, server.DEPTH_MAX_FRAMES)
        self.assertAlmostEqual(fps, server.DEPTH_MAX_FRAMES / 600, places=2)

    def test_sizes(self):
        import depth
        (mw, mh), (sw, sh) = depth.sizes(1920, 1080, 320)
        self.assertEqual((mw % 14, mh % 14), (0, 0))
        self.assertEqual(sw, 320)
        self.assertEqual(sh, 180)
        (mw, mh), (sw, sh) = depth.sizes(1080, 1920, 320)   # portrait
        self.assertGreater(mh, mw)
        self.assertEqual((sw, sh), (180, 320))

    def test_smooth(self):
        import depth
        v = depth.smooth([0, 0, 9, 0, 0], 1)
        self.assertEqual(list(v), [0, 3, 3, 3, 0])
        self.assertEqual(list(depth.smooth([1, 2], 0)), [1, 2])


@unittest.skipUnless(gpu_ready(), "no NVIDIA card or Depth Anything model here")
class DepthRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = server.make_server(PORT)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        server.depth_cancel_all()
        cls.srv.shutdown()

    def upload(self, path, name="clip.mp4"):
        with open(path, "rb") as fh:
            return call(f"/api/upload?name={urllib.parse.quote(name)}", fh.read())

    def test_a_build_then_cache(self):
        clip = make_clip(os.path.join(TMP, "walk.mp4"), seconds=2, size="640x360", fps=24, audio=False)
        up = self.upload(clip)
        st = call(f"/api/depth/status?id={up['id']}")
        self.assertEqual(st["state"], "none")
        self.assertEqual(st["frames"], 48)
        st = call("/api/depth/start", {"id": up["id"]})
        self.assertEqual(st["state"], "running")
        again = call("/api/depth/start", {"id": up["id"]})       # asking twice doesn't start a second run
        self.assertEqual(again["state"], "running")
        self.assertEqual(sum(1 for j in server.State.depth.values() if j.state == "running"), 1)
        st = wait(up["id"])
        self.assertEqual(st["state"], "ready", st)
        self.assertEqual((st["width"], st["height"], st["fps"]), (320, 180, 24.0))
        self.assertEqual(st["frames"], 48)
        data, headers = call(f"/api/depth/data?id={up['id']}", raw=True)
        self.assertEqual(len(data), 48 * 320 * 180)
        self.assertEqual(headers["X-Frames"], "48")
        self.assertEqual(min(data), 0)            # every frame is stretched from far (0) to near (255)
        self.assertEqual(max(data), 255)
        # the camera path was measured with the depth (the usual lens), and another lens is measured when asked
        base = server.depth_base(server.State.media[up["id"]])
        self.assertTrue(os.path.exists(base + ".path-fov65.json"))
        self.assertIn("path", json.load(open(base + ".json", encoding="utf-8")))
        p65 = call(f"/api/depth/path?id={up['id']}&fov=65")
        self.assertEqual((p65["frames"], len(p65["poses"]), len(p65["poses"][0]), p65["fov"]), (48, 48, 16, 65))
        self.assertEqual(p65["poses"][0], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        p80 = call(f"/api/depth/path?id={up['id']}&fov=80")
        self.assertEqual(p80["fov"], 80)
        self.assertTrue(os.path.exists(base + ".path-fov80.json"))
        # the same video opened again (a new copy, a new id) finds its depth at once
        up2 = self.upload(clip, "same video again.mp4")
        self.assertNotEqual(up2["id"], up["id"])
        self.assertEqual(call(f"/api/depth/status?id={up2['id']}")["state"], "ready")
        self.assertEqual(call("/api/depth/start", {"id": up2["id"]})["state"], "ready")
        job = server.State.depth[server.depth_base(server.State.media[up["id"]])]
        for _ in range(50):          # the depth is ready a moment before its process has quite finished
            if job.state == "ready":
                break
            time.sleep(0.1)
        self.assertEqual(job.state, "ready")
        logs = os.listdir(os.path.join(server.DATA_DIR, "logs"))
        self.assertFalse([f for f in logs if f.startswith("depth-")], "the log of a good run is removed")

    def test_b_cancel(self):
        clip = make_clip(os.path.join(TMP, "long.mp4"), seconds=40, size="640x360", fps=30, audio=False)
        up = self.upload(clip)
        call("/api/depth/start", {"id": up["id"]})
        time.sleep(1.0)
        st = call("/api/depth/cancel", {"id": up["id"]})
        self.assertEqual(st["state"], "cancelled")
        job = server.State.depth[server.depth_base(server.State.media[up["id"]])]
        job.proc.wait(10)
        self.assertIsNotNone(job.proc.poll(), "the depth process is gone")
        base = job.base
        self.assertFalse(os.path.exists(base + ".json"))
        # a fresh start after a stop works
        self.assertEqual(call("/api/depth/start", {"id": up["id"]})["state"], "running")
        call("/api/depth/cancel", {"id": up["id"]})

    def test_c_errors(self):
        self.assertIn("error", call("/api/depth/status?id=nope"))
        self.assertIn("error", call("/api/depth/path?id=nope"))
        clip = make_clip(os.path.join(TMP, "fresh.mp4"), seconds=1, size="320x240", fps=25, audio=False)
        up = self.upload(clip)
        r = call(f"/api/depth/data?id={up['id']}")
        self.assertEqual(r.get("status"), 400)
        self.assertIn("no AI depth", r.get("error", ""))
        r = call(f"/api/depth/path?id={up['id']}")
        self.assertEqual(r.get("status"), 400)
        self.assertIn("no AI depth", r.get("error", ""))

    def test_d_server_gone(self):
        """depth.py stops by itself when the server that started it disappears."""
        import subprocess
        clip = make_clip(os.path.join(TMP, "orphan.mp4"), seconds=30, size="640x360", fps=30, audio=False)
        stand_in = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                                    creationflags=server.NO_WINDOW)   # plays the server
        p = subprocess.Popen([server.python_exe(), "-u", os.path.join(APP, "depth.py"), "--src", clip, "--out",
                              os.path.join(TMP, "orphan"), "--parent", str(stand_in.pid)],
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             creationflags=server.NO_WINDOW)
        try:
            first = json.loads(p.stdout.readline())
            self.assertEqual(first["t"], "progress")
            stand_in.kill()
            self.assertEqual(p.wait(20), 3)
        finally:
            for proc in (p, stand_in):
                if proc.poll() is None:
                    proc.kill()
            p.stdout.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
