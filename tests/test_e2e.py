"""End-to-end test: drives the real app in Chrome and writes a review folder.

    python tests/test_e2e.py                      -> <your temp folder>\\Time Cube - Tests
    python tests/test_e2e.py --review <folder>    -> any folder
    python tests/test_e2e.py --clip <video>       -> your own footage (with a person in it) instead of the sample

Real footage is needed for AI depth, Track (it finds "person") and What the AI sees: by default that is 4 s of the
free sample clip samples/dancer.mp4 (see samples/CREDITS.md), with a generated tone as its sound.

Every feature is exercised through the real window: the four looks, the wire plane axes, the moves and the Tour
choreography, AI depth (build, stop, the cache, relief, trims, motion, fade, pivot, export with depth), the camera
path (measured, lens, made-up paths, map, export), object tracking (find, what's here, click, follow, isolate,
dim, track line, export, the cache), what the AI sees (file details, captions, a place guess, a question, the
caption on the picture and in an export), presets, mouse turning and zoom, every source (video, photos, webcam, demo), the time controls, the
MP4 export (fast WebCodecs path and raw fallback, with sound), transparent exports (ProRes 4444 .mov, VP9 alpha
.webm, transparent PNG), live recording, the dark / light window, formats, saved settings and stopping an export. Exported files are checked with ffprobe and copied into the review folder next to
screenshots and an HTML report. The unit tests (python + node) run first.
"""
import argparse
import glob
import html
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
DEFAULT_REVIEW = os.path.join(tempfile.gettempdir(), "Time Cube - Tests")
PORT = 47968
TMP = tempfile.mkdtemp(prefix="timecube-e2e-")
OUT = os.path.join(TMP, "out")
os.environ["TIME_CUBE_DATA"] = os.path.join(TMP, "data")
os.environ["TIME_CUBE_OUT"] = OUT
sys.path.insert(0, APP)
import server  # noqa: E402

DIRS = {"shots": "01 Screens", "looks": "02 Looks", "reel": "03 Tour moves", "presets": "04 Presets",
        "sources": "05 Sources", "exports": "06 Exports", "depth": "07 AI depth", "path": "08 Camera path",
        "track": "09 Track", "ai": "10 What the AI sees"}


# ---- results ------------------------------------------------------------------------------------------------
class Results:
    def __init__(self):
        self.items = []
        self.section = "General"

    def check(self, name, ok, detail=""):
        self.items.append({"section": self.section, "name": name, "ok": bool(ok), "detail": str(detail)})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
        return ok

    def media(self, kind, path, caption):
        self.items.append({"section": self.section, "media": kind, "path": path, "caption": caption})

    @property
    def passed(self):
        return sum(1 for i in self.items if i.get("ok") is True)

    @property
    def failed(self):
        return sum(1 for i in self.items if i.get("ok") is False)


R = Results()


def section(name):
    R.section = name
    print(f"\n== {name}", flush=True)


# ---- helpers ------------------------------------------------------------------------------------------------
def ffmpeg():
    return server.find_tool("ffmpeg")


def probe(path):
    r = subprocess.run([server.find_tool("ffprobe"), "-v", "error", "-count_frames", "-show_entries",
                        "stream=codec_type,codec_name,width,height,nb_read_frames,r_frame_rate,avg_frame_rate:"
                        "format=duration", "-of", "json", path], capture_output=True, text=True)
    d = json.loads(r.stdout or "{}")
    v = next((s for s in d.get("streams", []) if s["codec_type"] == "video"), {})
    a = next((s for s in d.get("streams", []) if s["codec_type"] == "audio"), None)
    return {"w": v.get("width"), "h": v.get("height"), "frames": int(v.get("nb_read_frames") or 0),
            "rate": v.get("r_frame_rate"), "avg": v.get("avg_frame_rate"), "codec": v.get("codec_name"),
            "audio": a is not None, "duration": float((d.get("format") or {}).get("duration") or 0)}


def alpha_frame(video, t, decoder=None):
    """One frame of a transparent video as an RGBA array (VP9 alpha needs the libvpx decoder)."""
    cmd = [ffmpeg(), "-v", "error"] + (["-c:v", decoder] if decoder else []) + [
        "-ss", f"{t:.3f}", "-i", video, "-frames:v", "1", "-pix_fmt", "rgba", "-f", "image2pipe", "-c:v", "png", "-"]
    import io
    return np.asarray(Image.open(io.BytesIO(subprocess.run(cmd, capture_output=True, check=True).stdout)).convert("RGBA"))


def over(rgba, colour=(230, 40, 170)):
    """An RGBA picture laid over a flat colour, to show what is see-through."""
    a = rgba[..., 3:4].astype(np.float32) / 255
    return (rgba[..., :3] * a + np.array(colour, np.float32) * (1 - a)).astype(np.uint8)


def pix_fmt(video):
    r = subprocess.run([server.find_tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=pix_fmt:stream_tags=alpha_mode", "-of", "json", video], capture_output=True, text=True)
    st = (json.loads(r.stdout or "{}").get("streams") or [{}])[0]
    return st.get("pix_fmt"), (st.get("tags") or {}).get("alpha_mode") or (st.get("tags") or {}).get("ALPHA_MODE")


def frame_of(video, t, dst):
    subprocess.run([ffmpeg(), "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", video, "-frames:v", "1", dst], check=True)
    return dst


def sample_clip(folder):
    """4 s of the sample clip (a dancer, free under the Pexels licence) with a 440 Hz tone: real footage with a person
    in it, for AI depth, Track and What the AI sees. None when the sample isn't there."""
    src = os.path.join(APP, "samples", "dancer.mp4")
    if not os.path.exists(src):
        return None
    os.makedirs(folder, exist_ok=True)
    out = os.path.join(folder, "dancer 4 s (Pexels).mp4")
    subprocess.run([ffmpeg(), "-v", "error", "-y", "-ss", "4", "-i", src, "-f", "lavfi", "-i",
                    "sine=frequency=440:sample_rate=48000", "-t", "4", "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out], check=True)
    return out


def make_fixtures(folder):
    """A 30 s 640x480 test clip with sound (long enough for the whole Tour loop), and four photos."""
    os.makedirs(folder, exist_ok=True)
    clip = os.path.join(folder, "fixture clip.mp4")
    subprocess.run([ffmpeg(), "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=640x480:rate=30:duration=30",
                    "-f", "lavfi", "-i", "sine=frequency=330:duration=30", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", clip], check=True)
    photos = []
    rng = np.random.default_rng(7)
    for i in range(4):
        yy, xx = np.mgrid[0:600, 0:800]
        base = np.array([[240, 200, 160], [60, 90, 200], [200, 60, 90], [40, 160, 90]][i], np.float32)
        img = base[None, None, :] * (0.6 + 0.4 * np.sin(xx / 60.0 + i) * np.cos(yy / 45.0))[..., None]
        cx, cy = 200 + 130 * i, 300
        img[(xx - cx) ** 2 + (yy - cy) ** 2 < 90 ** 2] = (20, 16, 14)
        img += rng.normal(0, 6, img.shape)
        p = os.path.join(folder, f"photo {i + 1}.png")
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(p)
        photos.append(p)
    return clip, photos


def stats(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    return a.std(), a


def differ(a, b):
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def psnr(a, b):
    mse = float(((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean())
    return 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def sheet(paths, dst, cols=6, w=270):
    ims = [Image.open(p).convert("RGB") for p in paths]
    if not ims:
        return None
    h = int(w * ims[0].height / ims[0].width)
    rows = (len(ims) + cols - 1) // cols
    s = Image.new("RGB", (w * min(cols, len(ims)), h * rows), "white")
    for i, im in enumerate(ims):
        s.paste(im.resize((w, h)), ((i % cols) * w, (i // cols) * h))
    s.save(dst)
    return dst


def newest(pattern, after=0):
    fs = [f for f in glob.glob(os.path.join(OUT, pattern)) if os.path.getmtime(f) > after and ".part." not in f]
    return max(fs, key=os.path.getmtime) if fs else None


# ---- the run ------------------------------------------------------------------------------------------------
def run(review, clip_arg):
    for d in DIRS.values():
        os.makedirs(os.path.join(review, d), exist_ok=True)
    rel = lambda p: os.path.relpath(p, review).replace("\\", "/")  # noqa: E731
    D = {k: os.path.join(review, v) for k, v in DIRS.items()}

    section("Unit tests")
    r = subprocess.run([sys.executable, os.path.join(HERE, "test_server.py")], capture_output=True, text=True, cwd=APP)
    tail = (r.stderr or "").strip().splitlines()
    R.check("server tests (upload, decode, encode, retries, keep-alive, saves)", r.returncode == 0,
            next((ln for ln in reversed(tail) if ln.startswith("Ran ")), "") + " " + (tail[-1] if tail else ""))
    r = subprocess.run([sys.executable, os.path.join(HERE, "test_depth.py")], capture_output=True, text=True, cwd=APP)
    tail = (r.stderr or "").strip().splitlines()
    R.check("AI depth tests (frame-rate plan, a real GPU run, cache, stop, server gone)", r.returncode == 0,
            next((ln for ln in reversed(tail) if ln.startswith("Ran ")), "") + " " + (tail[-1] if tail else ""))
    r = subprocess.run([sys.executable, os.path.join(HERE, "test_path.py")], capture_output=True, text=True, cwd=APP)
    tail = (r.stderr or "").strip().splitlines()
    errs = [ln.strip().split("...")[-1].strip() for ln in (r.stdout + chr(10) + r.stderr).splitlines()
            if "% of the path" in ln]
    R.check("camera path tests (a rendered scene with a known path: true depth and Depth Anything)", r.returncode == 0,
            " · ".join(errs) or (tail[-1] if tail else ""))
    r = subprocess.run([sys.executable, os.path.join(HERE, "test_track.py")], capture_output=True, text=True, cwd=APP)
    lines = [ln.strip().split("...")[-1].strip() for ln in (r.stdout + chr(10) + r.stderr).splitlines()
             if "follow: seen" in ln or "find 'red ball'" in ln]
    tail = (r.stderr or "").strip().splitlines()
    R.check("tracking tests (a rendered ball: find, click, follow, leaving the picture, the server)", r.returncode == 0,
            " · ".join(lines) or (tail[-1] if tail else ""))
    r = subprocess.run([sys.executable, os.path.join(HERE, "test_caption.py")], capture_output=True, text=True, cwd=APP)
    lines = [ln.strip() for ln in (r.stdout + chr(10) + r.stderr).splitlines() if "asked 'What colour" in ln]
    tail = (r.stderr or "").strip().splitlines()
    R.check("caption tests (the rendered ball described, its colour asked, the cache, a missing environment)",
            r.returncode == 0, " · ".join(lines) or (tail[-1] if tail else ""))
    r = subprocess.run(["node", os.path.join(HERE, "js", "camera.test.mjs")], capture_output=True, text=True, cwd=APP)
    R.check("choreography tests (keys, seamless loop, smoothness, look weights, sway)", r.returncode == 0,
            (r.stdout or r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr) else "")
    r = subprocess.run(["node", os.path.join(HERE, "js", "shape.test.mjs")], capture_output=True, text=True, cwd=APP)
    R.check("shape of time tests (stack, echo, spiral)", r.returncode == 0,
            (r.stdout or r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr) else "")

    clip, photos = make_fixtures(os.path.join(TMP, "fixtures"))
    srv = server.make_server(PORT)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True, args=[
            "--autoplay-policy=no-user-gesture-required", "--use-fake-device-for-media-stream",
            "--use-fake-ui-for-media-stream"])
        ctx = browser.new_context(viewport={"width": 1500, "height": 940})
        ctx.grant_permissions(["camera"], origin=f"http://127.0.0.1:{PORT}")
        pg = ctx.new_page()
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.on("console", lambda m: m.type == "error" and "blob:" not in m.text and
              errors.append(f"{m.text} {m.location.get('url', '')}"))
        pg.on("response", lambda r: r.status >= 400 and "/api/" in r.url and errors.append(f"{r.status} {r.url}"))
        ev = pg.evaluate

        def canvas_shot(path):
            pg.locator("#view").screenshot(path=path)
            return path

        def settle(t=0.6):
            time.sleep(t)

        def still(name, folder, set_=None, view=None, preset="classic", seek=None):
            ev("""([preset, set, view]) => { const tc = window.__tc; tc.applyPreset(preset);
                 for (const [k, v] of Object.entries(set || {})) tc.setParam(k, v);
                 Object.assign(tc.view, {yaw: 0, pitch: 0, zoom: 1, panX: 0, panY: 0}, view || {}); }""",
               [preset, set_ or {}, view or {}])
            if seek is not None:
                ev(f"window.__tc.seek({seek})")
            settle(0.5)
            return canvas_shot(os.path.join(folder, name + ".png"))

        # -- start ------------------------------------------------------------------------------------------
        section("Start")
        t0 = time.time()
        pg.goto(f"http://127.0.0.1:{PORT}/")
        pg.wait_for_function("window.__tc && window.__tc.src && window.__tc.src.kind === 'demo'", timeout=20000)
        R.check("the window opens with the demo playing", True, f"{time.time() - t0:.1f} s")
        R.check("the window is dark by default", ev("document.documentElement.dataset.theme") == "dark" and
                ev("getComputedStyle(document.body).backgroundColor") == "rgb(18, 18, 20)",
                ev("getComputedStyle(document.body).backgroundColor"))
        R.check("WebGL2 renderer and the cube volume exist",
                ev("!!window.__tc.renderer.gl && window.__tc.renderer.volN > 0"),
                ev("JSON.stringify([__tc.renderer.volW, __tc.renderer.volH, __tc.renderer.volN])"))
        n_ctl = ev("document.querySelectorAll('#controls .ctl').length")
        R.check("the side panel is built from the settings list", n_ctl >= 35, f"{n_ctl} controls")
        fps = ev("""() => new Promise(r => { let n = 0; const t0 = performance.now();
                 const f = () => { n++; performance.now() - t0 < 2000 ? requestAnimationFrame(f) : r(n / 2); };
                 requestAnimationFrame(f); })""")
        R.check("the preview runs smoothly", fps >= 30, f"{fps:.0f} frames a second (headless)")
        head0 = ev("__tc.renderer.head")
        settle(0.5)
        R.check("new frames enter the cube while playing", ev("__tc.renderer.head") != head0)
        pg.screenshot(path=os.path.join(D["shots"], "window.png"))
        R.media("img", rel(os.path.join(D["shots"], "window.png")), "The window (dark): sources, presets, format, export")
        pg.keyboard.press("t")
        settle(0.3)
        light = ev("document.documentElement.dataset.theme") == "light" and \
            ev("getComputedStyle(document.body).backgroundColor") == "rgb(230, 230, 228)"
        pg.screenshot(path=os.path.join(D["shots"], "window light.png"))
        pg.click("#theme")
        settle(0.3)
        R.check("T and the sun / moon button switch the window light and dark", light and
                ev("document.documentElement.dataset.theme") == "dark")
        R.media("img", rel(os.path.join(D["shots"], "window light.png")), "The same window in light mode")

        # -- looks ------------------------------------------------------------------------------------------
        section("Looks")
        shots = {}
        for m, name in [(0, "Wire"), (1, "Slices"), (2, "Solid"), (3, "Points")]:
            shots[name] = still(f"look {m} {name}", D["looks"], {"mode": "still", "autoMorph": False, "morph": m},
                                {"yaw": 38, "pitch": 18})
        arr = {k: stats(v)[1] for k, v in shots.items()}
        for k, v in shots.items():
            R.check(f"{k} draws a picture", stats(v)[0] > 8, f"contrast {stats(v)[0]:.1f}")
        R.check("the four looks are different", min(differ(arr[a], arr[b]) for a, b in [
            ("Wire", "Slices"), ("Slices", "Solid"), ("Wire", "Solid"), ("Solid", "Points"), ("Slices", "Points")]) > 3)
        pg.click(".looks button[data-look='2']")
        R.check("the Solid button sets the look", abs(ev("__tc.S.morph") - 2) < 1e-6 and not ev("__tc.S.autoMorph"))
        pg.keyboard.press("4")
        R.check("key 4 sets Points", abs(ev("__tc.S.morph") - 3) < 1e-6)
        pg.keyboard.press("1")
        R.check("key 1 sets Wire", abs(ev("__tc.S.morph")) < 1e-6)
        axes = {}
        for ax in ["columns", "frames", "rows"]:
            axes[ax] = still(f"wire planes through {ax}", D["looks"],
                             {"mode": "still", "autoMorph": False, "morph": 0, "wireAxis": ax}, {"yaw": 38, "pitch": 18})
        a2 = {k: stats(v)[1] for k, v in axes.items()}
        R.check("wire planes through Columns / Frames / Rows differ",
                differ(a2["columns"], a2["frames"]) > 2 and differ(a2["frames"], a2["rows"]) > 2)
        for extra, set_ in [("depth relief", {"wireWave": 1.2, "wireAxis": "frames"}),
                            ("glass slices clear brights", {"morph": 1, "sliceClear": 0.9, "sliceOpacity": 0.3}),
                            ("night background glow", {"background": "#0a0a0d", "glow": 1.2, "glowThreshold": 0.4}),
                            ("newest frame at the back", {"morph": 2, "newestBack": True})]:
            still(extra, D["looks"], {"mode": "still", "autoMorph": False, "morph": 0, **set_}, {"yaw": 38, "pitch": 18})
        looks = sorted(glob.glob(os.path.join(D["looks"], "*.png")))
        R.media("img", rel(sheet(looks, os.path.join(D["looks"], "_sheet.jpg"), cols=5)),
                "Wire, Slices, Solid, Points, the wire plane axes and a few finishes")
        # the shapes of time, with every look
        shapes = {}
        for shp in ["cube", "stack", "echo"]:
            for m, name in [(0, "wire"), (1, "slices"), (2, "solid"), (3, "points")]:
                shapes[(shp, m)] = stats(still(f"shape {shp} {m} {name}", D["looks"],
                                               {"mode": "still", "autoMorph": False, "morph": m, "shape": shp},
                                               {"yaw": 38, "pitch": 18}))[1]
        R.check("Stack and Echo reshape every look", min(differ(shapes[("cube", m)], shapes[(shp, m)])
                                                         for shp in ["stack", "echo"] for m in range(4)) > 2)
        spun = stats(still("shape cube spiral 180 solid", D["looks"], {"mode": "still", "autoMorph": False, "morph": 2,
                                                                        "spiral": 180}, {"yaw": 38, "pitch": 18}))[1]
        R.check("Spiral twists the block", differ(spun, shapes[("cube", 2)]) > 2)
        ev("__tc.setParam('spiral', 0); __tc.setParam('shape', 'cube')")
        shp_pngs = sorted(glob.glob(os.path.join(D["looks"], "shape *.png")))
        R.media("img", rel(sheet(shp_pngs, os.path.join(D["looks"], "_shapes sheet.jpg"), cols=4)),
                "Shape of time: Cube, Echo, Stack (rows, in that order) with Wire, Slices, Solid and Points; then"
                " the Cube with a 180° spiral")

        # -- video + Tour choreography -----------------------------------------------------------------------
        section("Video source and the Tour moves")
        pg.set_input_files("#fileVideo", clip)
        pg.wait_for_function("__tc.src && __tc.src.kind === 'video' && __tc.src.info", timeout=90000)
        info = ev("__tc.src.info")
        R.check("a video opens and the server keeps a copy", info["width"] == 640 and info["audio"],
                f"{info['width']}x{info['height']} {info['duration']} s, sound: {info['audio']}")
        R.check("the cube takes the video's shape", ev("Math.abs(__tc.renderer.volW / __tc.renderer.volH - 4/3) < 0.02"))
        R.check("the transport shows the clip length", pg.inner_text("#dur").strip() == "0:30.0", pg.inner_text("#dur"))
        settle(1.5)
        t_play = ev("__tc.sceneT")
        R.check("the video plays and drives the clock", t_play > 0.5, f"{t_play:.2f} s")
        pg.keyboard.press(" ")
        settle(0.3)
        R.check("Space pauses", ev("__tc.src.el.paused"))
        pg.keyboard.press(" ")
        sources = [clip] + ([clip_arg] if clip_arg else [])
        for si, src_path in enumerate(sources):
            if si:
                pg.set_input_files("#fileVideo", src_path)
                pg.wait_for_function("__tc.src && __tc.src.kind === 'video' && __tc.src.info", timeout=60000)
            ev("__tc.applyPreset('classic')")
            reel = []
            label = "fixture" if not si else os.path.splitext(os.path.basename(src_path))[0][:40]
            ev("__tc.restart()")
            ev("__tc.setPlaying(true)")
            start = time.time()
            for k, t in enumerate([0.3, 1.6, 3.2, 5.0, 6.8, 8.2, 9.6, 12.0, 15.0, 19.6, 21.0, 25.6]):
                while time.time() - start < t:
                    time.sleep(0.01)
                pth = os.path.join(D["reel"], f"{label} {k:02d} t{t:04.1f}.png")
                canvas_shot(pth)
                reel.append(pth)
            R.media("img", rel(sheet(reel, os.path.join(D["reel"], f"_{label} sheet.jpg"), cols=6)),
                    f"Tour moves on the {label} clip at 0.3, 1.6, 3.2, 5, 6.8, 8.2, 9.6, 12, 15, 19.6, 21 and 25.6 s")
        R.check("the Tour moves give a different picture through the loop",
                differ(stats(reel[1])[1], stats(reel[4])[1]) > 2)

        # -- mouse --------------------------------------------------------------------------------------------
        section("Mouse and keys")
        box = pg.locator("#view").bounding_box()
        cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        pg.mouse.move(cx, cy)
        pg.mouse.down()
        pg.mouse.move(cx + 120, cy + 40, steps=8)
        pg.mouse.up()
        v = ev("({...__tc.view})")
        R.check("dragging turns the cube", v["yaw"] > 30 and v["pitch"] > 5, f"yaw {v['yaw']:.0f}°, pitch {v['pitch']:.0f}°")
        pg.mouse.wheel(0, -400)
        settle(0.2)
        R.check("the wheel zooms", ev("__tc.view.zoom") > 1.2, f"{ev('__tc.view.zoom'):.2f}×")
        pg.keyboard.down("Shift")
        pg.mouse.move(cx, cy)
        pg.mouse.down()
        pg.mouse.move(cx + 60, cy, steps=4)
        pg.mouse.up()
        pg.keyboard.up("Shift")
        R.check("Shift-drag pans", ev("__tc.view.panX") > 0.05)
        pg.mouse.dblclick(cx, cy)
        settle(0.2)
        v = ev("({...__tc.view})")
        R.check("double-click resets the view", v["yaw"] == 0 and v["zoom"] == 1 and v["panX"] == 0)

        # -- presets ------------------------------------------------------------------------------------------
        section("Presets")
        names = ev("Array.from(document.querySelectorAll('#preset option')).map(o => o.value)")
        pres = []
        for pid in names:
            pg.select_option("#preset", pid)
            ev("__tc.setParam('mode', __tc.S.mode)")
            settle(0.8)
            pres.append(canvas_shot(os.path.join(D["presets"], f"preset {pid}.png")))
        R.check("every preset applies", len(pres) == len(names) >= 10, ", ".join(names))
        R.media("img", rel(sheet(pres, os.path.join(D["presets"], "_sheet.jpg"), cols=5)),
                f"The {len(names)} presets (the depth ones on the fixture clip use its depth only once it has it)")
        pg.select_option("#preset", "classic")
        ev("__tc.setParam('wireGrid', 60)")
        pg.click("#resetAll")
        R.check("Reset to preset undoes changes", ev("__tc.S.wireGrid") == 22, ev("__tc.S.wireGrid"))

        # -- time controls ------------------------------------------------------------------------------------
        section("Time controls")
        ev("__tc.applyPreset('classic'); __tc.setParam('history', 120)")
        R.check("Frames in the cube resizes the history", ev("__tc.renderer.volN") == 120)
        ev("__tc.setParam('quality', 512)")
        long_side = "Math.max(__tc.renderer.volW, __tc.renderer.volH)"   # the clip may be portrait
        R.check("Detail Best makes a volume 512 on its long side", ev(long_side) == 512, ev(long_side))
        ev("__tc.setParam('quality', 256)")
        R.check("Detail Draft makes a volume 256 on its long side", ev(long_side) == 256, ev(long_side))
        ev("__tc.applyPreset('classic')")

        # -- exports ------------------------------------------------------------------------------------------
        section("Export video")

        def export(label, set_in=None, set_out=None, fmt="reel", fps=30, length=None, raw=False, quality=18):
            ev("window.__lastExport = null")
            ev(f"window.__forceRaw = {str(raw).lower()}")
            if set_in is not None:
                ev(f"__tc.src.in = {set_in}; __tc.src.out = {set_out}")
            pg.click("#export")
            pg.select_option("#exFormat", fmt)
            pg.click(f"#exFps button[data-v='{fps}']")
            pg.click(f"#exQuality button[data-v='{quality}']")
            if length is not None:
                pg.fill("#exLen", str(length))
            pg.fill("#exName", label)
            t = time.time()
            pg.click("#exGo")
            pg.wait_for_function("window.__lastExport", timeout=600000)
            res = ev("window.__lastExport")
            took = time.time() - t
            pg.screenshot(path=os.path.join(D["shots"], f"export dialog {label}.png"))
            pg.click("#exCancel")
            return res, took

        res, took = export("video vertical 30fps", 1, 5)
        ok = R.check("video export finishes", bool(res.get("path")), res.get("error") or res.get("via"))
        if ok:
            pr = probe(res["path"])
            R.check("1080x1920 H.264, exactly 120 frames at 30 fps", pr["w"] == 1080 and pr["h"] == 1920 and
                    pr["codec"] == "h264" and pr["frames"] == 120 and pr["avg"] in ("30/1",), json.dumps(pr))
            R.check("the video's sound is in the export", pr["audio"] and abs(pr["duration"] - 4) < 0.1)
            R.check("export speed", True, f"{120 / took:.1f} frames a second ({res.get('via')})")
            dst = os.path.join(D["exports"], "video vertical 30fps.mp4")
            shutil.copy(res["path"], dst)
            R.media("video", rel(dst), "Vertical 1080×1920, 30 fps, fixture clip 1–5 s, with sound")
        res, took = export("video square raw path", 0, 2, fmt="square", raw=True, quality=14)
        ok = R.check("raw-pixel fallback export finishes", bool(res.get("path")), res.get("error") or res.get("via"))
        if ok:
            pr = probe(res["path"])
            R.check("1080x1080, 60 frames, sound", pr["w"] == 1080 and pr["h"] == 1080 and pr["frames"] == 60 and pr["audio"],
                    json.dumps(pr))
            shutil.copy(res["path"], os.path.join(D["exports"], "video square raw path.mp4"))
        # transparent: the checkerboard preview, ProRes 4444 and VP9 alpha
        ev("__tc.setParam('transparent', true)")
        settle(0.5)
        canvas_shot(os.path.join(D["shots"], "transparent preview.png"))
        R.media("img", rel(os.path.join(D["shots"], "transparent preview.png")),
                "Transparent background: the preview shows a checkerboard")
        alpha_sheet = []
        for kind, ext, codec, dec in [("prores", "mov", "prores", None), ("webm", "webm", "vp9", "libvpx-vp9")]:
            ev("window.__lastExport = null; window.__forceRaw = false")
            ev("__tc.src.in = 2; __tc.src.out = 4")
            pg.click("#export")
            R.check(f"the export dialog offers transparent video ({kind})",
                    pg.is_visible("#exAlphaRow") and "on" in (pg.get_attribute("#exBg button[data-v='transparent']", "class") or ""))
            pg.click(f"#exAlpha button[data-v='{kind}']")
            pg.select_option("#exFormat", "reel")
            pg.click("#exFps button[data-v='30']")
            pg.fill("#exName", f"transparent {kind}")
            t = time.time()
            pg.click("#exGo")
            pg.wait_for_function("window.__lastExport", timeout=600000)
            res = ev("window.__lastExport")
            took = time.time() - t
            pg.click("#exCancel")
            if not R.check(f"transparent {kind} export finishes", bool(res.get("path")), res.get("error") or res.get("via")):
                continue
            pr = probe(res["path"])
            fmt, mode = pix_fmt(res["path"])
            R.check(f"{ext}: {codec}, 1080x1920, 60 frames at 30 fps, with sound",
                    res["path"].endswith("." + ext) and pr["codec"] == codec and (pr["w"], pr["h"]) == (1080, 1920)
                    and pr["frames"] == 60 and pr["avg"] == "30/1" and pr["audio"], json.dumps(pr))
            R.check(f"{ext} carries an alpha channel", (fmt or "").startswith("yuva") or mode == "1",
                    f"pix_fmt {fmt}, alpha_mode {mode}")
            a = alpha_frame(res["path"], 1.0, dec)
            R.check(f"{ext}: the background is see-through and the cube is solid",
                    a[5, 5, 3] < 8 and a[..., 3].max() > 245 and (a[..., 3] < 8).mean() > 0.3,
                    f"corner alpha {a[5, 5, 3]}, max {a[..., 3].max()}, {100 * (a[..., 3] < 8).mean():.0f}% clear")
            R.check(f"{ext} export speed", True, f"{60 / took:.1f} frames a second")
            dst = os.path.join(D["exports"], f"transparent.{ext}")
            shutil.copy(res["path"], dst)
            pm = os.path.join(D["exports"], f"transparent {ext} over magenta.png")
            Image.fromarray(over(a)).save(pm)
            alpha_sheet.append(pm)
        R.media("img", rel(sheet(alpha_sheet, os.path.join(D["exports"], "_transparent sheet.jpg"), cols=2, w=405)),
                "A frame of the ProRes 4444 .mov and of the VP9 alpha .webm laid over magenta: only the cube and its glow")
        before = time.time()
        ev("__tc.snapPNG()")
        f = None
        for _ in range(100):
            f = newest("*.png", before)
            if f:
                break
            time.sleep(0.1)
        time.sleep(0.3)
        if R.check("a transparent PNG is saved", bool(f)):
            im = np.asarray(Image.open(f).convert("RGBA"))
            R.check("the PNG keeps its transparency", Image.open(f).mode == "RGBA" and im[5, 5, 3] == 0 and
                    im[..., 3].max() > 245, f"mode {Image.open(f).mode}, corner alpha {im[5, 5, 3]}")
            shutil.copy(f, os.path.join(D["exports"], "transparent still.png"))
        ev("__tc.setParam('transparent', false)")

        # stop an export halfway
        ev("window.__lastExport = null; window.__forceRaw = false; __tc.src.in = 0; __tc.src.out = 20")
        pg.click("#export")
        pg.select_option("#exFormat", "wide4k")
        pg.fill("#exName", "stopped export")
        pg.click("#exGo")
        pg.wait_for_function("document.getElementById('exStatus').textContent.startsWith('Frame 1')", timeout=60000)
        pg.click("#exCancel")
        pg.wait_for_function("window.__lastExport", timeout=60000)
        settle(0.5)
        left = glob.glob(os.path.join(OUT, "stopped export*"))
        R.check("Stop ends an export and leaves no file", ev("window.__lastExport.error") == "stopped" and not left,
                pg.inner_text("#exStatus"))
        pg.click("#exCancel")

        # -- PNG and recording ----------------------------------------------------------------------------------
        # -- AI depth -----------------------------------------------------------------------------------------
        section("AI depth")
        dshot = lambda name: canvas_shot(os.path.join(D["depth"], name + ".png"))  # noqa: E731

        def open_video(path):
            """Open a video and wait for it (not the one still showing) to be ready."""
            ev("window.__prevSrc = __tc.src")
            pg.set_input_files("#fileVideo", path)
            pg.wait_for_function("__tc.src !== window.__prevSrc && __tc.src.kind === 'video' && __tc.src.info &&"
                                 " __tc.depth.mid === __tc.src.info.id && !['wait', 'loading'].includes(__tc.depth.state)", timeout=90000)

        def look(set_, view=None, wait=0.5):
            ev("""([set, view]) => { const tc = window.__tc;
                 for (const [k, v] of Object.entries(set)) tc.setParam(k, v);
                 Object.assign(tc.view, {yaw: 0, pitch: 0, zoom: 1, panX: 0, panY: 0}, view || {}); }""", [set_, view or {}])
            settle(wait)

        # the fixture clip is open: no depth yet; start a run and stop it
        pg.wait_for_function("['none', 'error'].includes(__tc.depth.state)", timeout=60000)
        R.check("a fresh video offers Build AI depth with a time estimate",
                ev("__tc.depth.state") == "none" and pg.inner_text("#depthGo") == "Build AI depth" and
                "graphics card" in pg.inner_text("#depthStatus"), pg.inner_text("#depthStatus"))
        pg.click("#depthGo")
        pg.wait_for_function("__tc.depth.state === 'running'", timeout=20000)
        R.check("building shows Stop and a progress bar", pg.inner_text("#depthGo") == "Stop" and
                pg.is_visible("#depthBar"))
        pg.wait_for_function("(__tc.depth.status || {}).done > 0", timeout=300000)   # the model loads first
        chip = pg.is_visible("#depthChip") and "depth" in pg.inner_text("#depthChip").lower()
        R.check("the picture shows the progress", chip, pg.inner_text("#depthChip"))
        pg.locator("#stage").screenshot(path=os.path.join(D["depth"], "00 building.png"))
        pg.click("#depthGo")
        pg.wait_for_function("__tc.depth.state === 'cancelled'", timeout=20000)
        R.check("Stop ends the run", ev("__tc.depth.state") == "cancelled" and
                pg.inner_text("#depthGo") == "Build AI depth" and "frames" in pg.inner_text("#depthStatus") and
                "its frames" not in pg.inner_text("#depthStatus"), pg.inner_text("#depthStatus"))

        # real footage when given (--clip), else the fixture clip: build its depth
        depth_clip = clip_arg or clip
        if clip_arg:
            open_video(depth_clip)
        ev("__tc.applyPreset('classic')")
        if ev("__tc.depth.state") != "ready":
            t = time.time()
            pg.click("#depthGo")
            pg.wait_for_function("['ready', 'error'].includes(__tc.depth.state)", timeout=600000)
            built = time.time() - t
        else:
            built = 0
        st = ev("({state: __tc.depth.state, frames: __tc.depth.frames, fps: __tc.depth.fps, w: __tc.depth.w, h: __tc.depth.h,"
                " error: __tc.depth.error})")
        ok = R.check("AI depth builds for the video", st["state"] == "ready", json.dumps(st) +
                     (f", {built:.1f} s" if built else ", from the cache"))
        if ok:
            info = ev("__tc.src.info")
            R.check("one depth frame per video frame (up to 30 fps), 320 px on the long side",
                    abs(st["fps"] - min(30, info["fps"])) < 0.01 and max(st["w"], st["h"]) == 320 and
                    abs(st["frames"] - round(info["duration"] * st["fps"])) <= 2, json.dumps(st))
            if built:
                R.check("when it's ready, the cube uses it", ev("__tc.S.depthSrc") == "ai" and ev("__tc.S.relief") >= 0.8 and
                        "AI depth ready" in pg.inner_text("#toast"), pg.inner_text("#toast"))
            ev("__tc.setParam('depthSrc', 'ai')")
            R.check("the panel says it's ready", "AI depth ready" in pg.inner_text("#depthStatus"),
                    pg.inner_text("#depthStatus"))
            pg.locator("#panel").screenshot(path=os.path.join(D["depth"], "01 panel.png"))
            R.media("img", rel(os.path.join(D["depth"], "01 panel.png")), "The Depth section with the depth ready")
            def preset(pid):
                """A preset, held still, with two seconds of the video in the cube (a preset restarts the history)."""
                ev(f"__tc.applyPreset('{pid}'); __tc.setParam('mode', 'still'); __tc.setParam('grain', 0);"
                   "__tc.setPlaying(false); __tc.seek(Math.min(1, __tc.src.duration / 4))")
                settle(0.8)
                ev("__tc.setPlaying(true)")
                settle(2.2)
                ev("__tc.setPlaying(false)")
                settle(0.3)

            turned = {"yaw": 36, "pitch": 9}
            preset("depthPoints")
            look({"relief": 0}, turned)
            flat = stats(dshot("10 points flat"))[1]
            look({"relief": 1}, turned)
            lifted = stats(dshot("11 points with AI relief"))[1]
            R.check("Relief lifts every frame by its depth", differ(flat, lifted) > 3, f"{differ(flat, lifted):.1f}")
            look({"depthSrc": "bright"}, turned)
            bright = stats(dshot("12 points with brightness relief"))[1]
            R.check("AI depth and brightness give different relief", differ(lifted, bright) > 3,
                    f"{differ(lifted, bright):.1f}")
            trim0 = ev("__tc.S.trimFar")      # the preset's own trim
            look({"depthSrc": "ai", "trimFar": 0.6}, turned)
            trimmed = stats(dshot("13 trim far away"))[1]
            bg = np.array([11, 12, 15], np.float32)
            cover = lambda a: float((np.abs(a.astype(np.float32) - bg).sum(-1) > 30).mean())  # noqa: E731
            R.check("Trim far away removes the far part", cover(trimmed) < cover(lifted) * 0.85,
                    f"{cover(lifted):.2f} -> {cover(trimmed):.2f} of the picture")
            look({"trimFar": trim0, "trimDark": 0.6}, turned)
            dark = lambda a: float(((np.abs(a.astype(np.float32) - bg).sum(-1) > 30) &  # noqa: E731
                                    (a.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32) < 110)).mean())
            nodark = stats(dshot("14 remove dark"))[1]
            R.check("Remove dark pixels removes the dark part", dark(nodark) < dark(lifted) * 0.8,
                    f"dark parts {dark(lifted):.3f} -> {dark(nodark):.3f} of the picture")
            look({"trimDark": 0, "motion": 1}, turned)
            R.check("Highlight motion changes the picture", differ(stats(dshot("15 highlight motion"))[1], lifted) > 2)
            look({"motion": 0, "fadeOld": 0}, turned)
            nofade = stats(dshot("16 no fade"))[1]
            R.check("Fade older frames changes the picture", differ(nofade, lifted) > 2, f"{differ(nofade, lifted):.1f}")
            look({"fadeOld": 0.55, "pivot": "centre"}, turned)
            R.check("Turn around: newest frame vs centre", differ(stats(dshot("17 turned around the centre"))[1], lifted) > 2)
            look({"pivot": "newest", "morph": 1}, turned)
            dshot("18 sheets")
            look({"morph": 0, "wireAxis": "frames"}, turned)
            dshot("19 wire frames with relief")
            preset("depthSheets")
            look({}, turned)
            dshot("20 preset depth sheets")
            preset("motionTrails")
            look({}, turned)
            dshot("21 preset motion trails")
            ev("__tc.applyPreset('depthPoints'); __tc.setPlaying(true)")
            y0 = ev("__tc.lookState(__tc.sceneT, 0).yaw")
            settle(1.5)
            y1 = ev("__tc.lookState(__tc.sceneT, 0).yaw")
            R.check("Sway swings around the front", y0 != y1 and abs(y1) <= 38, f"yaw {y0:.1f}° -> {y1:.1f}°")
            sway = []
            for k in range(6):
                sway.append(dshot(f"3{k} sway {k}"))
                settle(0.9)
            R.media("img", rel(sheet(sorted(glob.glob(os.path.join(D["depth"], "1*.png"))) +
                                     sorted(glob.glob(os.path.join(D["depth"], "2*.png"))),
                                     os.path.join(D["depth"], "_looks sheet.jpg"), cols=6)),
                    "AI depth: Points flat / with AI relief / with brightness relief, trim far, remove dark, highlight"
                    " motion, no fade, turned around the centre, sheets, wire frames, the Depth sheets and Motion trails"
                    " presets")
            R.media("img", rel(sheet(sway, os.path.join(D["depth"], "_sway sheet.jpg"), cols=6)),
                    "The Depth points preset playing (Sway), about a second apart")

            # an export carries the depth: its first frame equals a still of the same moment
            ev("__tc.applyPreset('depthPoints'); __tc.setParam('mode', 'still'); __tc.setParam('grain', 0);"
               "Object.assign(__tc.view, {yaw: 30, pitch: 8}); __tc.setPlaying(false);"
               "__tc.src.in = Math.min(1, __tc.src.duration / 4); __tc.src.out = __tc.src.in + 2; __tc.seek(0)")
            settle(1.2)
            pg.select_option("#format", "reel")
            before = time.time()
            ev("__tc.snapPNG()")
            snap = None
            for _ in range(100):
                snap = newest("*.png", before)
                if snap:
                    break
                time.sleep(0.1)
            time.sleep(0.3)
            res, took = export("ai depth points", fmt="reel", fps=30)
            ok = R.check("an export with AI depth finishes", bool(res.get("path")), res.get("error") or res.get("via"))
            if ok:
                pr = probe(res["path"])
                R.check("1080x1920, 60 frames (2 s at 30 fps)", pr["w"] == 1080 and pr["frames"] == 60, json.dumps(pr))
                if snap:
                    f0 = frame_of(res["path"], 0, os.path.join(TMP, "depth f0.png"))
                    q = psnr(np.asarray(Image.open(f0).convert("RGB")), np.asarray(Image.open(snap).convert("RGB")))
                    R.check("the export's first frame matches the preview still (depth included)", q > 28,
                            f"PSNR {q:.1f} dB")
                dst = os.path.join(D["exports"], "ai depth points.mp4")
                shutil.copy(res["path"], dst)
                R.media("video", rel(dst), "Export with AI depth: the Depth points preset, 2 s")

            # -- the camera path, on the same video ------------------------------------------------------
            section("Camera path")
            pshot = lambda name: canvas_shot(os.path.join(D["path"], name + ".png"))  # noqa: E731
            ev("document.getElementById('toast').classList.remove('show')")   # not over the pictures
            R.check("the Camera path section only shows in that layout",
                    ev("document.querySelector('.sec[data-sec=path]').classList.contains('hidden')"))
            ev("__tc.applyPreset('cameraPath'); __tc.setParam('depthSrc', 'ai'); __tc.setParam('grain', 0)")
            pg.wait_for_function("['ready', 'error'].includes(__tc.camPath.state)", timeout=120000)
            cp = ev("({state: __tc.camPath.state, steps: __tc.camPath.steps, solved: __tc.camPath.solved, error: __tc.camPath.error,"
                    " frames: (__tc.camPath.data || {}).frames})")
            R.check("the camera path is measured from the video", cp["state"] == "ready" and cp["solved"] >= 0.8 * cp["steps"]
                    and cp["frames"] == st["frames"], pg.inner_text("#pathStatus"))
            R.check("Camera path shows its section and hides the Solid one",
                    not ev("document.querySelector('.sec[data-sec=path]').classList.contains('hidden')") and
                    ev("document.querySelector('.sec[data-sec=solid]').classList.contains('hidden')"))
            ev("__tc.setParam('mode', 'still'); __tc.setPlaying(false); __tc.seek(Math.min(1, __tc.src.duration / 4))")
            settle(0.8)
            ev("__tc.setPlaying(true)")
            settle(2.4)
            ev("__tc.setPlaying(false)")
            settle(0.4)
            look({}, {})
            front = stats(pshot("10 front"))[1]
            look({}, {"yaw": 35, "pitch": 10})
            side = stats(pshot("11 turned"))[1]
            R.check("turning shows the frames in 3D along the path", differ(front, side) > 4, f"{differ(front, side):.1f}")
            look({}, {"pitch": 55, "zoom": 0.6})
            pshot("12 from above")
            look({"layout": "cube"}, {"yaw": 35, "pitch": 10})
            cube = stats(pshot("13 the same in the time cube"))[1]
            R.check("Camera path and Time cube give different pictures", differ(side, cube) > 4)
            look({"layout": "path", "showPath": False}, {"yaw": 35, "pitch": 10})
            R.check("Show the path draws the path and the outline of now",
                    differ(stats(pshot("14 without the path line"))[1], side) > 0.3)
            look({"showPath": True, "morph": 1, "sliceOpacity": 0.5}, {"yaw": 35, "pitch": 10})
            pshot("15 sheets")
            look({"morph": 0}, {"yaw": 35, "pitch": 10})
            pshot("16 wire")
            # the map
            pg.locator("#panel").screenshot(path=os.path.join(D["path"], "01 panel.png"))
            mp = np.asarray(Image.open(os.path.join(D["path"], "01 panel.png")).convert("RGB")).astype(np.float32)
            box = pg.locator("#pathMap").bounding_box()
            pb = pg.locator("#panel").bounding_box()
            crop = mp[int(box["y"] - pb["y"]):int(box["y"] - pb["y"] + box["height"]),
                      int(box["x"] - pb["x"]):int(box["x"] - pb["x"] + box["width"])]
            orange = ((crop[..., 0] > 200) & (crop[..., 1] > 120) & (crop[..., 1] < 190) & (crop[..., 2] < 120)).mean()
            grey = ((np.abs(crop[..., 0] - crop[..., 2]) < 12) & (crop[..., 0] > 110)).mean()
            R.check("the map shows the path and where the camera is", orange > 0.0005 and grey > 0.002,
                    f"{orange * 100:.2f} % orange (now and the stretch in the cube), {grey * 100:.2f} % grey (the path)")
            R.media("img", rel(os.path.join(D["path"], "01 panel.png")), "The Camera path section: the map and the path")
            # another lens: measured again
            ev("__tc.setParam('camFov', 80)")
            pg.wait_for_function("__tc.camPath.fov === 80 && ['ready', 'error'].includes(__tc.camPath.state)", timeout=120000)
            R.check("a new Camera lens measures the path again", ev("__tc.camPath.state") == "ready" and
                    "lens 80°" in pg.inner_text("#pathStatus"), pg.inner_text("#pathStatus"))
            ev("__tc.setParam('camFov', 65)")
            pg.wait_for_function("__tc.camPath.fov === 65 && __tc.camPath.state === 'ready'", timeout=120000)
            # made-up paths
            made = {}
            for kind in ["orbit", "walk", "turn"]:
                look({"pathFrom": kind, "morph": 3}, {"yaw": 25, "pitch": 12})
                made[kind] = stats(pshot(f"2{len(made)} made-up {kind}"))[1]
                m = ev(f"Array.from(__tc.poseFor(3))")
                R.check(f"{kind.capitalize()} is a made-up path that moves", any(abs(m[i] - (1 if i % 5 == 0 else 0)) > 1e-3
                                                                                  for i in range(16)))
            R.check("Orbit, Walk and Turn differ", min(differ(made["orbit"], made["walk"]), differ(made["walk"], made["turn"]),
                                                       differ(made["orbit"], made["turn"])) > 1.5)
            for pid in ["walkThrough", "orbitRing"]:
                ev(f"__tc.applyPreset('{pid}'); __tc.setParam('mode', 'still'); __tc.setParam('grain', 0); __tc.setPlaying(true)")
                settle(3.0)
                ev("__tc.setPlaying(false)")
                look({}, {"yaw": 25, "pitch": 12})
                pshot(f"3{int(pid == 'orbitRing')} preset {pid}")
            R.media("img", rel(sheet(sorted(glob.glob(os.path.join(D["path"], "1*.png"))) +
                                     sorted(glob.glob(os.path.join(D["path"], "2*.png"))) +
                                     sorted(glob.glob(os.path.join(D["path"], "3*.png"))),
                                     os.path.join(D["path"], "_sheet.jpg"), cols=5)),
                    "Camera path: front (the video), turned, from above, the same in the time cube, without the path"
                    " line, sheets, wire; Orbit, Walk, Turn; the Walk through time and Orbit ring presets")
            # an export along the camera path matches the preview
            ev("__tc.applyPreset('cameraPath'); __tc.setParam('mode', 'still'); __tc.setParam('grain', 0);"
               "Object.assign(__tc.view, {yaw: 25, pitch: 8}); __tc.setPlaying(false);"
               "__tc.src.in = Math.min(1, __tc.src.duration / 4); __tc.src.out = __tc.src.in + 2; __tc.seek(0)")
            pg.wait_for_function("__tc.camPath.state === 'ready'", timeout=60000)
            settle(1.2)
            before = time.time()
            ev("__tc.snapPNG()")
            snap = None
            for _ in range(100):
                snap = newest("*.png", before)
                if snap:
                    break
                time.sleep(0.1)
            time.sleep(0.3)
            res, took = export("camera path", fmt="reel", fps=30)
            ok = R.check("an export along the camera path finishes", bool(res.get("path")), res.get("error") or res.get("via"))
            if ok and snap:
                f0 = frame_of(res["path"], 0, os.path.join(TMP, "path f0.png"))
                q = psnr(np.asarray(Image.open(f0).convert("RGB")), np.asarray(Image.open(snap).convert("RGB")))
                R.check("its first frame matches the preview still", q > 28, f"PSNR {q:.1f} dB")
                dst = os.path.join(D["exports"], "camera path.mp4")
                shutil.copy(res["path"], dst)
                R.media("video", rel(dst), "Export along the camera path: the Camera path preset, 2 s")
            ev("__tc.applyPreset('classic')")

            # -- following an object, on the same video ----------------------------------------------
            section("Track")
            tshot = lambda name: canvas_shot(os.path.join(D["track"], name + ".png"))  # noqa: E731
            ev("document.getElementById('toast').classList.remove('show'); __tc.applyPreset('classic');"
               "__tc.setParam('depthSrc', 'ai'); __tc.setPlaying(false); __tc.seek(Math.min(2, __tc.src.duration / 2))")
            settle(1.0)
            R.check("the Track section asks to pause and find something",
                    "Pause where" in pg.inner_text("#trackStatus"), pg.inner_text("#trackStatus"))
            pg.fill("#trackText", "person")
            t = time.time()
            pg.click("#trackFind")
            pg.wait_for_function("__tc.track.busy === '' && (__tc.track.cands.length > 0 || !!__tc.track.error)", timeout=300000)
            cands = ev("__tc.track.cands.map(c => ({label: c.label, area: c.area, score: c.score}))")
            R.check("Find marks what was typed, in the picker", len(cands) >= 1 and ev("__tc.picker.isOpen"),
                    f"{len(cands)} found in {time.time() - t:.1f} s: {cands}")
            pg.locator("#stage").screenshot(path=os.path.join(D["track"], "01 picker after find.png"))
            R.media("img", rel(os.path.join(D["track"], "01 picker after find.png")),
                    "Find “person”: the paused frame with what was found outlined")
            pg.click("#trackHere")
            pg.wait_for_function("__tc.track.busy === ''", timeout=120000)
            labels = ev("__tc.track.labels.map(l => l.label)")
            R.check("What's in this moment names things as chips", len(labels) >= 2 and
                    ev("document.querySelectorAll('#trackChips .chip').length") >= len(labels), ", ".join(labels))
            # a click on the picture cuts out what is under it
            if cands:
                bx = ev("__tc.track.cands[0].box")
                cb = pg.locator("#pickerCanvas").bounding_box()
                pg.mouse.click(cb["x"] + cb["width"] * (bx[0] + bx[2]) / 2, cb["y"] + cb["height"] * (bx[1] * 0.4 + bx[3] * 0.6))
                pg.wait_for_function("__tc.track.busy === ''", timeout=120000)
                picked = ev("__tc.track.cands[__tc.track.sel]")
                R.check("a click in the picture picks an outline or cuts out what is there", picked is not None,
                        f"{picked['label'] if picked else None}")
                ev("(() => { const i = __tc.track.cands.findIndex(c => c.label === 'person'); if (i >= 0) __tc.picker.onPick(i); })()")
            pg.locator("#panel").screenshot(path=os.path.join(D["track"], "02 panel found.png"))
            t = time.time()
            pg.click("#pickerGo")
            pg.wait_for_function("__tc.track.state === 'following'", timeout=60000)
            settle(1.0)
            R.check("Follow it shows the progress and Stop", pg.inner_text("#trackGo") == "Stop" and pg.is_visible("#trackBar"),
                    pg.inner_text("#trackStatus"))
            pg.wait_for_function("['ready', 'error', 'picked', 'none'].includes(__tc.track.state)", timeout=1500000)
            tr = ev("({state: __tc.track.state, seen: __tc.track.seen, frames: __tc.track.frames, error: __tc.track.error})")
            took = time.time() - t
            followed = R.check("the object is followed through the video",
                               tr["state"] == "ready" and tr["seen"] >= 0.6 * tr["frames"],
                               f"{pg.inner_text('#trackStatus')} ({took:.0f} s, {1000 * took / max(1, tr['frames']):.0f} ms a frame)")
            if followed:
                pg.locator("#panel").screenshot(path=os.path.join(D["track"], "03 panel followed.png"))
                R.media("img", rel(os.path.join(D["track"], "03 panel followed.png")), "The Track section once followed")
                ev("document.getElementById('toast').classList.remove('show')")

                def held(preset, extra=None, view=None):
                    ev(f"__tc.applyPreset('{preset}'); __tc.setParam('depthSrc', 'ai'); __tc.setParam('mode', 'still');"
                       "__tc.setParam('grain', 0); __tc.setPlaying(false); __tc.seek(Math.min(1, __tc.src.duration / 4))")
                    settle(0.8)
                    ev("""(e) => { for (const [k, v] of Object.entries(e)) __tc.setParam(k, v); __tc.setPlaying(true); }""",
                       extra or {})
                    settle(2.4)
                    ev("__tc.setPlaying(false)")
                    look({}, view or {"yaw": 35, "pitch": 10})

                held("followOne", {"isolate": False, "dimRest": 0, "trackLine": False})
                plain = stats(tshot("10 no track effects"))[1]
                look({"trackLine": True}, {"yaw": 35, "pitch": 10})
                line = stats(tshot("11 track line"))[1]
                changed = float((np.abs(line.astype(np.float32) - plain.astype(np.float32)).max(-1) > 60).mean())
                R.check("the track line is drawn through the frames", changed > 0.0003,
                        f"{changed * 100:.2f} % of the picture changed by the line")
                look({"dimRest": 0.8}, {"yaw": 35, "pitch": 10})
                dim = stats(tshot("12 dim the rest"))[1]
                R.check("Dim the rest fades everything but the object", dim.mean() < plain.mean() - 3,
                        f"brightness {plain.mean():.1f} -> {dim.mean():.1f}")
                look({"isolate": True}, {"yaw": 35, "pitch": 10})
                iso = stats(tshot("13 isolate tracked"))[1]
                R.check("Isolate tracked keeps only the object", cover(iso) < cover(plain) * 0.8,
                        f"{cover(plain):.2f} -> {cover(iso):.2f} of the picture")
                look({}, {})
                tshot("14 isolate front")
                held("followPath")
                tshot("15 follow along the path")
                held("slit", {"isolate": True})
                tshot("16 solid isolated")
                R.media("img", rel(sheet(sorted(glob.glob(os.path.join(D["track"], "1*.png"))),
                                         os.path.join(D["track"], "_sheet.jpg"), cols=4)),
                        "Track: no effects, the track line, Dim the rest, Isolate tracked (turned and front), Follow along"
                        " the path, the Solid block isolated")
                # an export with the object isolated matches the preview
                ev("__tc.applyPreset('followOne'); __tc.setParam('mode', 'still'); __tc.setParam('grain', 0);"
                   "Object.assign(__tc.view, {yaw: 30, pitch: 8}); __tc.setPlaying(false);"
                   "__tc.src.in = Math.min(1, __tc.src.duration / 4); __tc.src.out = __tc.src.in + 2; __tc.seek(0)")
                settle(1.2)
                before = time.time()
                ev("__tc.snapPNG()")
                snap = None
                for _ in range(100):
                    snap = newest("*.png", before)
                    if snap:
                        break
                    time.sleep(0.1)
                time.sleep(0.3)
                res, took = export("follow one", fmt="reel", fps=30)
                ok2 = R.check("an export with the object isolated finishes", bool(res.get("path")), res.get("error") or res.get("via"))
                if ok2 and snap:
                    f0 = frame_of(res["path"], 0, os.path.join(TMP, "track f0.png"))
                    q = psnr(np.asarray(Image.open(f0).convert("RGB")), np.asarray(Image.open(snap).convert("RGB")))
                    R.check("its first frame matches the preview still", q > 28, f"PSNR {q:.1f} dB")
                    dst = os.path.join(D["exports"], "follow one.mp4")
                    shutil.copy(res["path"], dst)
                    R.media("video", rel(dst), "Export with the followed object isolated (Follow one), 2 s")
            ev("__tc.applyPreset('classic')")

            # -- what the AI sees, on the same video ---------------------------------------------------
            section("What the AI sees")
            ashot = lambda name: canvas_shot(os.path.join(D["ai"], name + ".png"))  # noqa: E731
            ev("document.getElementById('toast').classList.remove('show'); __tc.applyPreset('classic');"
               "__tc.setParam('mode', 'still'); __tc.setParam('grain', 0); __tc.setPlaying(false)")
            facts = pg.inner_text("#aiFacts")
            R.check("the file's details are listed", os.path.basename(depth_clip) in facts and "fps" in facts,
                    facts.replace(chr(10), " · "))
            described = False
            for attempt in range(4):          # another program can be holding the graphics card: wait for it
                t = time.time()
                pg.click("#aiDescribe")
                pg.wait_for_function("['ready', 'error'].includes(__tc.ai.state)", timeout=1200000)
                if ev("__tc.ai.state") == "ready" or "graphics card has only" not in ev("__tc.ai.error"):
                    break
                print("  the graphics card is busy:", ev("__tc.ai.error"), flush=True)
                time.sleep(90)
            moments = ev("__tc.ai.moments")
            described = R.check("the video is described, moment by moment", ev("__tc.ai.state") == "ready" and
                                len(moments) >= 3 and all(len(m["text"]) > 15 for m in moments),
                                f"{time.time() - t:.0f} s: " + " | ".join(f"{m['time']} s {m['text']}" for m in moments)
                                if moments else ev("__tc.ai.error"))
            if described:
                R.check("with a guess at the place", len(ev("__tc.ai.place")) > 5, ev("__tc.ai.place"))
                pg.locator(".sec[data-sec=ai]").screenshot(path=os.path.join(D["ai"], "01 section.png"))
                R.media("img", rel(os.path.join(D["ai"], "01 section.png")), "What the AI sees, described")
                pg.click("#aiMoments li:last-child button")
                settle(0.8)
                R.check("clicking a moment goes there", abs(ev("__tc.src.el.currentTime") - moments[-1]["time"]) < 0.2,
                        f"{ev('__tc.src.el.currentTime'):.2f} s")
                pg.fill("#aiQuestion", "What colour is the sky?")
                t = time.time()
                pg.click("#aiAsk")
                pg.wait_for_function("!__tc.ai.asking", timeout=600000)
                ans = ev("__tc.ai.answer")
                R.check("a question about the moment is answered", ans and not ans.startswith("Couldn't"),
                        f"{ans!r} ({time.time() - t:.1f} s)")
                look({"captionOn": False}, {})
                plain = stats(ashot("10 no caption"))[1]
                look({"captionOn": True}, {})
                capd = stats(ashot("11 caption on the picture"))[1]
                low = slice(int(capd.shape[0] * 0.7), None)
                R.check("the caption shows on the picture", differ(plain[low], capd[low]) > 2,
                        f"{differ(plain[low], capd[low]):.1f} in the lower part")
                R.media("img", rel(os.path.join(D["ai"], "11 caption on the picture.png")), "Caption on the picture")
                # an export carries the caption: its first frame matches the preview
                ev("__tc.setParam('captionOn', true); Object.assign(__tc.view, {yaw: 20, pitch: 6}); __tc.setPlaying(false);"
                   "__tc.src.in = Math.max(0, __tc.src.duration - 2.2); __tc.src.out = __tc.src.in + 2; __tc.seek(0)")
                settle(1.2)
                before = time.time()
                ev("__tc.snapPNG()")
                snap = None
                for _ in range(100):
                    snap = newest("*.png", before)
                    if snap:
                        break
                    time.sleep(0.1)
                time.sleep(0.3)
                res, took = export("with a caption", fmt="reel", fps=30)
                ok3 = R.check("an export with the caption finishes", bool(res.get("path")), res.get("error") or res.get("via"))
                if ok3 and snap:
                    f0 = frame_of(res["path"], 0, os.path.join(TMP, "caption f0.png"))
                    q = psnr(np.asarray(Image.open(f0).convert("RGB")), np.asarray(Image.open(snap).convert("RGB")))
                    R.check("its first frame matches the preview still (caption included)", q > 28, f"PSNR {q:.1f} dB")
                    dst = os.path.join(D["exports"], "with a caption.mp4")
                    shutil.copy(res["path"], dst)
                    R.media("video", rel(dst), "Export with the caption on the picture, 2 s")
            ev("__tc.applyPreset('classic'); __tc.setParam('captionOn', false)")

            # the same video opened again finds its depth at once
            section("AI depth, again")
            t = time.time()
            open_video(depth_clip)
            pg.wait_for_function("__tc.depth.state !== 'loading'", timeout=60000)
            R.check("the same video opened again has its depth at once (a new copy, found by its contents)",
                    ev("__tc.depth.state") == "ready" and not ev("__tc.depth.built"),
                    f"{ev('__tc.depth.state')} {time.time() - t:.1f} s after opening")
            if followed:
                try:
                    pg.wait_for_function("__tc.track.state === 'ready'", timeout=60000)
                except Exception:
                    pass
                R.check("... and its followed object too", ev("__tc.track.state") == "ready", ev("__tc.track.state"))
                pg.click("#trackClear")
                pg.wait_for_function("__tc.track.state === 'none'", timeout=30000)
                R.check("Clear the track removes it", ev("__tc.track.state") == "none" and
                        not ev("__tc.lookState(0, 0).track.line"))
            if described:
                try:
                    pg.wait_for_function("__tc.ai.state === 'ready'", timeout=60000)
                except Exception:
                    pass
                R.check("... and what the AI saw", ev("__tc.ai.state") == "ready" and len(ev("__tc.ai.moments")) >= 3)
            ev("__tc.applyPreset('classic')")

        section("PNG still and live recording")
        fmt_now = ev("document.getElementById('format').value")
        want = {"reel": (1080, 1920), "portrait": (1080, 1350), "square": (1080, 1080), "wide": (1920, 1080),
                "wide4k": (3840, 2160)}[fmt_now]
        before = time.time()
        pg.keyboard.press("p")
        for _ in range(100):
            f = newest("*.png", before)
            if f:
                break
            time.sleep(0.1)
        time.sleep(0.3)
        ok = R.check("P saves a PNG of this frame", bool(f), os.path.basename(f or ""))
        if ok:
            im = Image.open(f)
            R.check("the PNG has the format's full size", im.size == want, f"{im.size} for {fmt_now}")
            shutil.copy(f, os.path.join(D["exports"], "still.png"))
            R.media("img", rel(os.path.join(D["exports"], "still.png")), "PNG still, full size")
        before = time.time()
        pg.click("#record")
        try:   # it starts once the canvas has drawn twice at full size
            pg.wait_for_function("document.getElementById('record').classList.contains('recording')", timeout=5000)
        except Exception:
            pass
        R.check("Record starts", ev("document.getElementById('record').classList.contains('recording')"))
        time.sleep(3.2)
        pg.click("#record")
        f = None
        try:
            pg.wait_for_function("document.getElementById('toast').textContent.includes('live')", timeout=120000)
            f = newest("*live*.mp4", before)
        except Exception:
            pass
        ok = R.check("the recording is saved as an MP4", bool(f), os.path.basename(f or ""))
        if ok:
            pr = probe(f)
            k = min(1, 1920 / max(want))   # live recordings are capped at 1920 on the long side
            rec_size = (round(want[0] * k / 2) * 2, round(want[1] * k / 2) * 2)
            R.check("the recording is full size (up to 1920), constant 30 fps, about 3 s", (pr["w"], pr["h"]) == rec_size and
                    pr["avg"] == "30/1" and 2.0 < pr["duration"] < 5.0, json.dumps(pr))
            R.check("the recording keeps its frames (H.264 recorder)", pr["frames"] >= 0.8 * 30 * min(pr["duration"], 3.2),
                    f"{pr['frames']} frames in {pr['duration']:.1f} s")
            shutil.copy(f, os.path.join(D["exports"], "live recording.mp4"))
            R.media("video", rel(os.path.join(D["exports"], "live recording.mp4")), "Record: 3 s live, as seen")

        # -- photos -------------------------------------------------------------------------------------------
        section("Photos, webcam and demo")
        pg.set_input_files("#filePhotos", photos)
        pg.wait_for_function("__tc.src && __tc.src.kind === 'photos'", timeout=20000)
        R.check("photos open as stop-motion", ev("__tc.src.images.length") == 4)
        R.check("photos use brightness depth (AI depth is for videos)", ev("__tc.depth.state") == "na" and
                "for videos" in pg.inner_text("#depthStatus") and not pg.is_visible("#depthGo"))
        R.check("tracking is for videos too", "for videos" in pg.inner_text("#trackStatus") and
                ev("document.getElementById('trackFind').disabled"))
        R.check("what the AI sees is for videos too", "for videos" in pg.inner_text("#aiStatus") and
                ev("document.getElementById('aiAsk').disabled"))
        R.check("the Photo hold control shows for photos",
                ev("!document.querySelector('.ctl[data-id=photoHold]').classList.contains('hidden')"))
        settle(1.5)
        still("photos solid", D["sources"], {"mode": "still", "autoMorph": False, "morph": 2}, {"yaw": 40, "pitch": -20})
        R.media("img", rel(os.path.join(D["sources"], "photos solid.png")), "Four photos as stop-motion, solid block")
        res, took = export("photos sequence", fmt="portrait")
        ok = R.check("photo export finishes", bool(res.get("path")), res.get("error") or res.get("via"))
        if ok:
            pr = probe(res["path"])
            R.check("1080x1350, the whole sequence (4 photos x 8 frames = 32 frames)",
                    pr["w"] == 1080 and pr["h"] == 1350 and pr["frames"] == 32, json.dumps(pr))
            shutil.copy(res["path"], os.path.join(D["exports"], "photos sequence.mp4"))

        pg.click("#openCamera")
        pg.wait_for_function("__tc.src && __tc.src.kind === 'camera'", timeout=20000)
        settle(1.5)
        R.check("the webcam opens (Chrome's test camera) and is mirrored", ev("__tc.S.mirror"),
                f"{ev('__tc.src.width')}x{ev('__tc.src.height')}")
        R.check("Export is replaced by Record for the webcam", ev("document.getElementById('export').disabled"))
        canvas_shot(os.path.join(D["sources"], "webcam.png"))
        R.media("img", rel(os.path.join(D["sources"], "webcam.png")), "Webcam (Chrome's fake test camera)")

        pg.click("#openDemo")
        pg.wait_for_function("__tc.src && __tc.src.kind === 'demo'", timeout=20000)
        # export matches the preview: frame 0 of an export equals a still of the same moment
        pg.select_option("#format", "reel")
        ev("__tc.applyPreset('classic'); __tc.setParam('mode', 'still'); __tc.setParam('autoMorph', false);"
           "__tc.setParam('morph', 2); __tc.setParam('grain', 0); Object.assign(__tc.view, {yaw: 30, pitch: -15});"
           "__tc.setPlaying(false); __tc.seek(0)")
        before = time.time()
        ev("__tc.snapPNG()")
        snap = None
        for _ in range(100):
            snap = newest("*.png", before)
            if snap:
                break
            time.sleep(0.1)
        time.sleep(0.3)
        res, took = export("demo 60fps", fmt="reel", fps=60, length=2)
        ok = R.check("demo export at 60 fps finishes", bool(res.get("path")), res.get("error") or res.get("via"))
        if ok and snap:
            pr = probe(res["path"])
            R.check("120 frames at exactly 60 fps", pr["frames"] == 120 and pr["avg"] == "60/1", json.dumps(pr))
            f0 = frame_of(res["path"], 0, os.path.join(TMP, "f0.png"))
            q = psnr(np.asarray(Image.open(f0).convert("RGB")), np.asarray(Image.open(snap).convert("RGB")))
            R.check("the export's first frame matches the preview still", q > 30, f"PSNR {q:.1f} dB")
            shutil.copy(res["path"], os.path.join(D["exports"], "demo 60fps.mp4"))

        # -- formats and saved settings ------------------------------------------------------------------------
        section("Formats and saved settings")
        for fid, ar in [("square", 1.0), ("wide", 16 / 9), ("reel", 9 / 16)]:
            pg.select_option("#format", fid)
            settle(0.3)
            b = pg.locator("#view").bounding_box()
            R.check(f"format {fid} shapes the preview", abs(b["width"] / b["height"] - ar) < 0.02,
                    f"{b['width']:.0f}x{b['height']:.0f}")
        ev("__tc.setParam('wireGrid', 31); __tc.setParam('background', '#123456')")
        settle(0.5)
        pg.reload()
        pg.wait_for_function("window.__tc && window.__tc.src", timeout=20000)
        R.check("settings are kept after closing", ev("__tc.S.wireGrid") == 31 and ev("__tc.S.background") == "#123456")
        ev("__tc.applyPreset('classic')")
        R.check("no errors in the window", not errors, "; ".join(errors[:3]))
        try:
            ctx.close()
            browser.close()
        except Exception as e:  # a slow browser shutdown must not cost the report
            print("browser close:", e)
    srv.shutdown()


def write_report(review, seconds):
    rows, sec = [], None
    for it in R.items:
        if it["section"] != sec:
            sec = it["section"]
            rows.append(f"<h2>{html.escape(sec)}</h2>")
        if "media" in it:
            if it["media"] == "img":
                rows.append(f'<figure><a href="{it["path"]}"><img src="{it["path"]}" loading="lazy"></a>'
                            f'<figcaption>{html.escape(it["caption"])}</figcaption></figure>')
            else:
                rows.append(f'<figure><video src="{it["path"]}" controls muted loop></video>'
                            f'<figcaption>{html.escape(it["caption"])}</figcaption></figure>')
        else:
            mark = "pass" if it["ok"] else "fail"
            rows.append(f'<div class="row {mark}"><b>{"✓" if it["ok"] else "✗"}</b> {html.escape(it["name"])}'
                        f'<span>{html.escape(it["detail"])}</span></div>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Time Cube tests</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
:root {{ --bg:#f3f3f1; --ink:#1c1b19; --muted:#74726c; --ok:#1f7a3a; --bad:#b3261e; --line:#dddcd7; }}
body {{ background:var(--bg); color:var(--ink); font:14px/1.45 "Segoe UI",system-ui,sans-serif; margin:0; padding:24px 28px 60px; }}
h1 {{ margin:0 0 4px; font-size:22px; }} h2 {{ margin:28px 0 8px; font-size:15px; border-bottom:1px solid var(--line); padding-bottom:4px; }}
.sum {{ color:var(--muted); margin-bottom:10px; }} .row {{ padding:3px 0; }} .row b {{ display:inline-block; width:18px; }}
.row.pass b {{ color:var(--ok); }} .row.fail {{ color:var(--bad); }} .row span {{ color:var(--muted); margin-left:8px; font-size:12.5px; word-break:break-all; }}
figure {{ margin:10px 0 16px; }} img {{ max-width:100%; border:1px solid var(--line); background:#fff; }}
video {{ max-height:520px; max-width:100%; border:1px solid var(--line); background:#000; }}
figcaption {{ color:var(--muted); font-size:12.5px; margin-top:4px; }}
</style></head><body>
<h1>Time Cube: end-to-end test</h1>
<div class="sum">{R.passed} passed, {R.failed} failed · {time.strftime('%Y-%m-%d %H:%M')} · {seconds:.0f} s ·
app in {html.escape(APP)}</div>
{''.join(rows)}
</body></html>"""
    path = os.path.join(review, "report.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(page)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", default=DEFAULT_REVIEW)
    ap.add_argument("--clip", default=None, help="your own footage (with a person in it) for AI depth, Track and What "
                    "the AI sees; by default 4 s of samples/dancer.mp4")
    a = ap.parse_args()
    if os.path.isdir(a.review):
        for d in DIRS.values():
            shutil.rmtree(os.path.join(a.review, d), ignore_errors=True)
    os.makedirs(a.review, exist_ok=True)
    t0 = time.time()
    try:
        run(a.review, a.clip or sample_clip(os.path.join(TMP, "sample")))
    except Exception:
        R.check("the test ran to the end", False, traceback.format_exc()[-800:])
    path = write_report(a.review, time.time() - t0)
    print(f"\n{R.passed} passed, {R.failed} failed -> {path}")
    shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(1 if R.failed else 0)


if __name__ == "__main__":
    main()
