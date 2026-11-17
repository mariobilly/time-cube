"""What the AI sees: captions and answers through the server, on a rendered clip whose content is known.

    python tests/test_caption.py     (needs the caption environment, the Qwen3-VL 4B file and the NVIDIA card)

The clip is the red ball rolling across a wall (from test_track). The captions should talk about a red ball, and
"What colour is the ball?" should be answered "red". Also: the moments are spread over the clip, the captions are
kept (the same video opened again has them), a question waits while the video is being described, and without the
caption environment the window gets a clear message.
"""
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
TMP = tempfile.mkdtemp(prefix="timecube-caption-test-")
os.environ["TIME_CUBE_DATA"] = os.path.join(TMP, "data")
os.environ["TIME_CUBE_OUT"] = os.path.join(TMP, "out")
sys.path.insert(0, APP)
sys.path.insert(0, HERE)
import server  # noqa: E402
import test_server  # noqa: E402
from test_server import call  # noqa: E402
from test_track import FPS, H, N, W, render  # noqa: E402

PORT = 47978
test_server.PORT = PORT


def ready():
    try:
        import caption
        import torch
        return torch.cuda.is_available() and caption.model_path() and os.path.exists(server.caption_python())
    except Exception:
        return False


@unittest.skipUnless(ready(), "no caption environment, Qwen3-VL 4B or NVIDIA card here")
class CaptionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clip = os.path.join(TMP, "ball.mp4")
        p = subprocess.Popen([server.find_tool("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                              "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0", "-c:v", "libx264", "-crf", "14",
                              "-pix_fmt", "yuv420p", cls.clip], stdin=subprocess.PIPE)
        p.stdin.write(np.array([render(i)[0] for i in range(N)]).tobytes())
        p.stdin.close()
        assert p.wait() == 0
        cls.srv = server.make_server(PORT)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        with open(cls.clip, "rb") as fh:
            cls.up = call("/api/upload?name=ball.mp4", fh.read())

    @classmethod
    def tearDownClass(cls):
        server.CAPTIONER.kill()
        cls.srv.shutdown()

    def test_a_describe_and_ask(self):
        mid = self.up["id"]
        self.assertEqual(call(f"/api/caption/status?id={mid}")["state"], "none")
        t = time.time()
        st = call("/api/caption/describe", {"id": mid})
        self.assertEqual(st["state"], "running")
        busy = call("/api/caption/ask", {"id": mid, "time": 1, "question": "What is this?"})
        self.assertIn("being described", busy.get("error", ""))
        end = time.time() + 300
        while time.time() < end and st["state"] == "running":
            time.sleep(0.5)
            st = call(f"/api/caption/status?id={mid}")
        took = time.time() - t
        self.assertEqual(st["state"], "ready", st)
        texts = [m["text"] for m in st["moments"]]
        print(f"\n  described in {took:.1f} s (with loading the model):", flush=True)
        for m in st["moments"]:
            print(f"    {m['time']} s: {m['text']}", flush=True)
        print(f"    place: {st['place']}", flush=True)
        self.assertEqual(len(texts), 3)          # a 4 s clip
        times = [m["time"] for m in st["moments"]]
        self.assertTrue(times[0] < 0.5 and times[-1] > N / FPS - 0.5, times)
        self.assertTrue(any(("ball" in x.lower() or "sphere" in x.lower()) and "red" in x.lower() for x in texts), texts)
        t = time.time()
        a = call("/api/caption/ask", {"id": mid, "time": 1.0, "question": "What colour is the ball?"})
        print(f"  asked 'What colour is the ball?': {a.get('answer')!r} ({time.time() - t:.1f} s)", flush=True)
        self.assertIn("red", a["answer"].lower())
        self.assertIn("error", call("/api/caption/ask", {"id": mid, "time": 1, "question": "  "}))
        # kept: the same video opened again has them
        with open(self.clip, "rb") as fh:
            up2 = call(f"/api/upload?name={urllib.parse.quote('ball again.mp4')}", fh.read())
        again = call(f"/api/caption/status?id={up2['id']}")
        self.assertEqual((again["state"], [m["text"] for m in again["moments"]]), ("ready", texts))
        self.assertEqual(call("/api/caption/clear", {"id": up2["id"]})["state"], "none")

    def test_b_no_environment(self):
        old = os.environ.get("TIME_CUBE_CAPTION_PYTHON")
        os.environ["TIME_CUBE_CAPTION_PYTHON"] = os.path.join(TMP, "no such python.exe")
        try:
            r = call("/api/caption/describe", {"id": self.up["id"]})
            self.assertEqual(r.get("status"), 400)
            self.assertIn("make_caption_env", r.get("error", ""))
        finally:
            if old is None:
                os.environ.pop("TIME_CUBE_CAPTION_PYTHON", None)
            else:
                os.environ["TIME_CUBE_CAPTION_PYTHON"] = old
            with server.State.lock:
                server.State.describes.clear()

    def test_c_times(self):
        self.assertEqual(len(server.caption_times(4)), 3)
        self.assertEqual(len(server.caption_times(12)), 5)
        self.assertEqual(len(server.caption_times(600)), 8)
        self.assertTrue(all(0 <= t <= 12 for t in server.caption_times(12)))


class ModelPathTest(unittest.TestCase):
    """Where the Qwen3-VL 4B file is looked for (no graphics card needed)."""

    def test_model_path(self):
        import caption
        folder = os.path.join(TMP, "qwen")
        os.makedirs(folder, exist_ok=True)
        f = os.path.join(folder, caption.MODEL_FILE)
        open(f, "wb").close()
        old_env, old_dir = os.environ.get(caption.MODEL_ENV), caption.MODEL_DIR
        try:
            caption.MODEL_DIR = os.path.join(TMP, "empty models")
            os.environ.pop(caption.MODEL_ENV, None)
            self.assertIsNone(caption.model_path())                 # nowhere
            os.environ[caption.MODEL_ENV] = f                        # the variable names the file
            self.assertEqual(caption.model_path(), f)
            os.environ[caption.MODEL_ENV] = folder                   # ... or a folder that holds it
            self.assertEqual(caption.model_path(), f)
            os.environ[caption.MODEL_ENV] = os.path.join(TMP, "nowhere.safetensors")
            self.assertIsNone(caption.model_path())
            caption.MODEL_DIR = folder                               # the app's models folder
            self.assertEqual(caption.model_path(), f)
        finally:
            caption.MODEL_DIR = old_dir
            if old_env is None:
                os.environ.pop(caption.MODEL_ENV, None)
            else:
                os.environ[caption.MODEL_ENV] = old_env


if __name__ == "__main__":
    unittest.main(verbosity=2)
