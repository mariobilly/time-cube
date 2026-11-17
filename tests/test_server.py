"""Server tests: upload + probe, frame-exact decode sessions, MP4 encode sessions with sound, saves and guards.

    python -m unittest tests.test_server      (from the app folder)
"""
import glob
import http.client
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
TMP = tempfile.mkdtemp(prefix="timecube-test-")
os.environ["TIME_CUBE_DATA"] = os.path.join(TMP, "data")
os.environ["TIME_CUBE_OUT"] = os.path.join(TMP, "out")
sys.path.insert(0, APP)
import server  # noqa: E402

PORT = 47969


def make_clip(path, seconds=2, size="320x180", fps=25, audio=True):
    ff = server.find_tool("ffmpeg")
    cmd = [ff, "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={fps}:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", path]
    subprocess.run(cmd, check=True)
    return path


def call(path, body=None, method=None, raw=False):
    data = body if isinstance(body, (bytes, bytearray)) or body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data, method=method or ("POST" if data is not None else "GET"))
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            out = r.read()
            return (out, dict(r.headers)) if raw else json.loads(out)
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return {"status": e.code, **json.loads(body or b"{}")}
        except ValueError:
            return {"status": e.code}


def probe(path):
    r = subprocess.run([server.find_tool("ffprobe"), "-v", "error", "-count_frames", "-show_entries",
                        "stream=codec_type,codec_name,width,height,nb_read_frames:format=duration", "-of", "json", path],
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = server.make_server(PORT)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.clip = make_clip(os.path.join(TMP, "clip.mp4"))
        with open(cls.clip, "rb") as fh:
            cls.up = call("/api/upload?name=my%20clip.mp4", fh.read())

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_ping(self):
        r = call("/api/ping")
        self.assertEqual(r["app"], "time-cube")
        self.assertTrue(r["ffmpeg"])

    def test_static(self):
        out, headers = call("/", raw=True)
        self.assertIn(b"Time Cube", out)
        self.assertEqual(call("/../server.py").get("status"), 404)

    def test_upload_probe(self):
        u = self.up
        self.assertEqual((u["width"], u["height"]), (320, 180))
        self.assertAlmostEqual(u["fps"], 25, places=1)
        self.assertAlmostEqual(u["duration"], 2, delta=0.1)
        self.assertTrue(u["audio"])
        bad = call("/api/upload?name=x.mp4", b"not a video at all")
        self.assertEqual(bad["status"], 400)

    def test_decode_exact_frames(self):
        d = call("/api/decode/open", {"id": self.up["id"], "fps": 30, "width": 160, "height": 90, "start": 0.5, "duration": 1})
        self.assertEqual(d["frames"], 30)
        got, done = 0, False
        while not done:
            data, headers = call(f"/api/decode/{d['sid']}/frames?n=7", raw=True)
            n = int(headers["X-Frames"])
            self.assertEqual(len(data), n * 160 * 90 * 4)
            got += n
            done = headers["X-Done"] == "1" or n == 0
        call(f"/api/decode/{d['sid']}/close", b"")
        self.assertEqual(got, 30)

    def test_encode_with_audio(self):
        w, h, fps = 64, 48, 30
        e = call("/api/encode/open", {"width": w, "height": h, "fps": fps, "name": "enc test",
                                      "audio": {"id": self.up["id"], "start": 0, "duration": 1}})
        self.assertTrue(e["audio"])
        frame = bytes([200, 30, 30, 255]) * (w * h)
        for _ in range(3):
            r = call(f"/api/encode/{e['sid']}/frame", frame * 10)
        self.assertEqual(r["frames"], 30)
        f = call(f"/api/encode/{e['sid']}/finish", b"")
        self.assertTrue(os.path.exists(f["path"]))
        self.assertTrue(f["path"].endswith("enc test.mp4"))
        info = probe(f["path"])
        v = next(s for s in info["streams"] if s["codec_type"] == "video")
        self.assertEqual((v["codec_name"], v["width"], v["height"], int(v["nb_read_frames"])), ("h264", w, h, 30))
        self.assertTrue(any(s["codec_type"] == "audio" for s in info["streams"]))
        self.assertFalse(os.path.exists(f["path"][:-4] + ".part.mp4"))
        # the same name again never overwrites
        e2 = call("/api/encode/open", {"width": w, "height": h, "fps": fps, "name": "enc test"})
        self.assertTrue(e2["path"].endswith("enc test (2).mp4"))
        call(f"/api/encode/{e2['sid']}/close", b"")
        self.assertFalse(os.path.exists(e2["path"]))

    def test_transparent_exports(self):
        import numpy as np
        w, h = 64, 48
        f = np.zeros((h, w, 4), np.uint8)
        f[4:20, 16:48] = (200, 40, 40, 255)        # an opaque box near the bottom (rows come bottom-up)
        f[30:36, :, :] = (40, 200, 40, 128)          # and a half see-through strip near the top
        frames = f.tobytes() * 10
        for kind, ext, codec, pix, decoder in [("prores", "mov", "prores", "yuva444p10le", None),
                                               ("webm", "webm", "vp9", "yuva420p", "libvpx-vp9")]:
            e = call("/api/encode/open", {"width": w, "height": h, "fps": 30, "name": f"alpha {kind}", "alpha": kind,
                                          "audio": {"id": self.up["id"], "start": 0, "duration": 10 / 30}})
            self.assertEqual(e["encoder"], kind)
            self.assertTrue(e["path"].endswith(f"alpha {kind}.{ext}"), e)
            call(f"/api/encode/{e['sid']}/frame?i=0", frames)
            r = call(f"/api/encode/{e['sid']}/finish", b"")
            self.assertEqual(r["frames"], 10)
            pr = subprocess.run([server.find_tool("ffprobe"), "-v", "error", "-show_entries",
                                 "stream=codec_type,codec_name,pix_fmt", "-of", "json", r["path"]],
                                capture_output=True, text=True, check=True)
            streams = json.loads(pr.stdout)["streams"]
            v = next(s for s in streams if s["codec_type"] == "video")
            self.assertEqual(v["codec_name"], codec)
            if kind == "prores":
                self.assertTrue(v["pix_fmt"].startswith("yuva444"), v["pix_fmt"])  # 4:4:4 with alpha
            self.assertTrue(any(s["codec_type"] == "audio" for s in streams), kind)
            cmd = [server.find_tool("ffmpeg"), "-v", "error"] + (["-c:v", decoder] if decoder else []) + [
                "-i", r["path"], "-frames:v", "1", "-pix_fmt", "rgba", "-f", "rawvideo", "-"]
            out = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.uint8).reshape(h, w, 4)
            self.assertLess(out[2, 2, 3], 8, f"{kind}: the empty corner must stay transparent")
            self.assertGreater(out[h - 1 - 12, 30, 3], 245, f"{kind}: the box must stay opaque (rows flipped back)")
            self.assertLess(out[8, 30, 3], 8, f"{kind}: rows must be flipped back to top-down")
            self.assertTrue(100 < out[h - 1 - 32, 4, 3] < 160, f"{kind}: half see-through stays half")
        self.assertEqual(call("/api/encode/open", {"width": 64, "height": 48, "fps": 30, "alpha": "gif"})["status"], 400)

    def test_retries_are_safe(self):
        w, h = 32, 32
        e = call("/api/encode/open", {"width": w, "height": h, "fps": 30, "name": "retry"})
        f = bytes(w * h * 4)
        self.assertEqual(call(f"/api/encode/{e['sid']}/frame?i=0", f)["frames"], 1)
        self.assertEqual(call(f"/api/encode/{e['sid']}/frame?i=0", f)["frames"], 1)   # resent: ignored
        self.assertEqual(call(f"/api/encode/{e['sid']}/frame?i=1", f)["frames"], 2)
        self.assertEqual(call(f"/api/encode/{e['sid']}/frame?i=5", f)["status"], 400)  # a gap is refused
        call(f"/api/encode/{e['sid']}/close", b"")
        d = call("/api/decode/open", {"id": self.up["id"], "fps": 10, "width": 32, "height": 18})
        a, ha = call(f"/api/decode/{d['sid']}/frames?n=3&from=0", raw=True)
        b, hb = call(f"/api/decode/{d['sid']}/frames?n=3&from=0", raw=True)   # the same batch again
        self.assertEqual(a, b)
        c, hc = call(f"/api/decode/{d['sid']}/frames?n=3&from=3", raw=True)
        self.assertEqual(int(hc["X-Frames"]), 3)
        self.assertNotEqual(a, c)
        self.assertEqual(call(f"/api/decode/{d['sid']}/frames?n=3&from=1")["status"], 400)
        call(f"/api/decode/{d['sid']}/close", b"")

    def test_bad_frames(self):
        e = call("/api/encode/open", {"width": 32, "height": 32, "fps": 30, "name": "bad"})
        self.assertEqual(call(f"/api/encode/{e['sid']}/frame", b"\0" * 100)["status"], 400)
        call(f"/api/encode/{e['sid']}/close", b"")
        self.assertEqual(call("/api/encode/open", {"width": 33, "height": 32, "fps": 30})["status"], 400)
        self.assertEqual(call("/api/decode/open", {"id": "nope", "fps": 30, "width": 32, "height": 32})["status"], 400)

    def test_save_png_and_reveal_guard(self):
        png = (b"\x89PNG\r\n\x1a\n" + b"\0" * 64)
        r = call("/api/save?ext=png&name=still", png)
        self.assertTrue(os.path.exists(r["path"]))
        self.assertEqual(call("/api/save?ext=exe&name=x", b"zz")["status"], 400)
        self.assertEqual(call("/api/reveal", {"path": r"C:\Windows\notepad.exe"})["status"], 400)

    def test_keep_alive_survives_errors(self):
        # one connection, like Chrome reusing it: bodies must never leak into the next request
        c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=30)
        seq = [("POST", "/api/heartbeat", "{}", 200), ("POST", "/api/heartbeat", "{}", 200),
               ("POST", "/api/nope", "xyz", 404), ("POST", "/api/save?ext=exe", "abcdef", 400),
               ("POST", "/api/encode/feedbeef/frame", "x" * 5000, 400), ("GET", "/api/ping", None, 200),
               ("GET", "/missing.js", None, 404), ("GET", "/api/ping", None, 200)]
        for method, path, body, status in seq:
            c.request(method, path, body=body)
            r = c.getresponse()
            r.read()
            self.assertEqual(r.status, status, path)
        c.close()

    def test_recording_to_mp4(self):
        webm = os.path.join(TMP, "rec.webm")
        subprocess.run([server.find_tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
                        "testsrc2=size=320x240:rate=24:duration=1.5", "-c:v", "libvpx-vp9", "-deadline", "realtime",
                        webm], check=True)
        with open(webm, "rb") as fh:
            r = call("/api/save?ext=mp4&name=rec%20live&w=320&h=240&fps=30", fh.read())
        self.assertTrue(r["path"].endswith("rec live.mp4"), r)
        info = probe(r["path"])
        v = next(s for s in info["streams"] if s["codec_type"] == "video")
        self.assertEqual((v["width"], v["height"]), (320, 240))
        self.assertAlmostEqual(int(v["nb_read_frames"]), 45, delta=2)   # 1.5 s at a constant 30 fps
        self.assertFalse(glob.glob(os.path.join(os.environ["TIME_CUBE_OUT"], "*.part.*")))
        # a recording ffmpeg can't read is kept as it came, never lost
        r = call("/api/save?ext=mp4&name=broken%20live&w=320&h=240&fps=30", b"this is not a webm file at all")
        self.assertTrue(r["path"].endswith(".webm") and os.path.getsize(r["path"]) > 0, r)
        self.assertFalse(os.path.exists(r["path"][:-5] + ".mp4"))

    def test_safe_stem(self):
        self.assertEqual(server.safe_stem('..\\a/b<c>:"d'), "bcd")
        self.assertEqual(server.safe_stem("CON"), "_CON")
        self.assertEqual(server.safe_stem(""), "time cube")


if __name__ == "__main__":
    unittest.main()
