"""Time Cube - local server (Python standard library + ffmpeg).

The cube itself is drawn in the window with WebGL2. This server does the jobs a browser can't do well:
  * keeps a copy of the video you open, so an export can read it frame by frame (ffmpeg decode sessions);
  * turns the frames the window renders into an H.264 MP4 with the source video's sound (ffmpeg encode sessions);
  * saves PNG stills and webcam recordings into Videos\\Time Cube (or the folder in TIME_CUBE_OUT);
  * gives every frame of a video AI depth: depth.py runs Depth Anything on the GPU in a process of its own, and the
    result is kept (named by the video's contents), so the same video opened again has its depth at once;
  * finds, cuts out and follows an object through a video: track.py, kept running while it is used;
  * says what the AI sees in a video and answers questions about a moment: caption.py (Qwen3-VL 4B, the single-file
    version ComfyUI uses, in its own small Python environment).
Nothing leaves this computer.
"""
import hashlib
import json
import mimetypes
import os
import queue
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

APP_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ID = "time-cube"
APP_NAME = "Time Cube"
PORT = int(os.environ.get("TIME_CUBE_PORT") or 47963)
STATIC = os.path.join(APP_DIR, "static")
DATA_DIR = os.environ.get("TIME_CUBE_DATA") or os.path.join(os.environ.get("LOCALAPPDATA", APP_DIR), "TimeCube")
MEDIA_DIR = os.path.join(DATA_DIR, "media")
DEPTH_DIR = os.path.join(DATA_DIR, "depth")
DEPTH_VERSION = 1             # depth.py's output format; cached depth of another version is not reused
DEPTH_LONG = 320              # stored depth: long side in pixels
DEPTH_MAX_FPS = 30
DEPTH_MAX_FRAMES = 2700       # 90 s at 30 fps; a longer video gets its depth at a lower frame rate
PATH_TIMEOUT = 600            # seconds a camera path may take to measure
TRACK_IDLE = 300              # a GPU worker closes after this many idle seconds (its graphics memory is freed)
# captions run in their own environment (the system's packages plus a newer transformers: make_caption_env.ps1)
CAPTION_ENV = os.path.join(os.environ.get("LOCALAPPDATA", APP_DIR), "TimeCube", "caption-env")
MAX_UPLOAD = 16 * 1024 ** 3   # a source video can be large; it is streamed to disk, never held in memory
MAX_FRAME = 8192 * 8192 * 4
MEDIA_KEEP_DAYS = 3           # copies of opened videos are removed after this many days
SESSION_IDLE = 90             # an export session nobody talks to for this long is closed
# Seconds without a heartbeat before an auto-exit server quits. Chrome slows timers in a minimised window to about
# one a minute, so this must stay well above 60 s.
IDLE_EXIT = 150
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")


def output_dir():
    """Videos\\Time Cube in your user folder, or the folder in the TIME_CUBE_OUT environment variable."""
    env = os.environ.get("TIME_CUBE_OUT")
    if env:
        return env
    return os.path.join(os.path.expanduser("~"), "Videos", APP_NAME)


def find_tool(name):
    exe = shutil.which(name)
    if exe:
        return exe
    root = os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Microsoft\WinGet\Packages")
    if os.path.isdir(root):
        for dirpath, _dirs, files in os.walk(root):
            if f"{name}.exe" in files:
                return os.path.join(dirpath, f"{name}.exe")
    return None


def no_throttle(pid=None):
    """Opt a process (default: this one) out of Windows power throttling (EcoQoS).

    Windows quietly slows down busy processes that have no window of their own (pythonw, ffmpeg pipes): after a few
    seconds a frame takes 8-10x longer. Returns True when the opt-out was applied.
    """
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes as wt

        class _PPT(ctypes.Structure):
            _fields_ = [("Version", wt.ULONG), ("ControlMask", wt.ULONG), ("StateMask", wt.ULONG)]

        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = wt.HANDLE
        k.OpenProcess.restype = wt.HANDLE
        k.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
        k.SetProcessInformation.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
        k.SetProcessInformation.restype = wt.BOOL
        k.CloseHandle.argtypes = [wt.HANDLE]
        h = k.GetCurrentProcess() if pid is None else k.OpenProcess(0x0200 | 0x1000, False, pid)
        if not h:
            return False
        st = _PPT(1, 0x1, 0x0)  # execution-speed throttling: controlled by us, and off
        ok = bool(k.SetProcessInformation(h, 4, ctypes.byref(st), ctypes.sizeof(st)))  # 4 = ProcessPowerThrottling
        if pid is not None:
            k.CloseHandle(h)
        return ok
    except Exception:
        return False


def safe_stem(name, fallback="time cube"):
    """A file name stem that is safe on Windows: no path parts, no reserved characters, never empty."""
    name = os.path.basename(str(name or "").replace("\\", "/"))
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name).strip(" .")
    name = re.sub(r"\s+", " ", name)[:120].strip(" .")
    if name.upper() in {"CON", "PRN", "AUX", "NUL"} or re.fullmatch(r"(?i)(COM|LPT)\d", name or ""):
        name = "_" + name
    return name or fallback


def unique_path(folder, stem, ext):
    """folder/stem.ext, or 'stem (2).ext', 'stem (3).ext'... so existing files are never overwritten."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{stem}.{ext}")
    n = 2
    while os.path.exists(path):
        path = os.path.join(folder, f"{stem} ({n}).{ext}")
        n += 1
    return path


def reserve(folder, stem, ext):
    """Like unique_path, but creates the file at once so two saves at the same moment never pick the same name."""
    while True:
        path = unique_path(folder, stem, ext)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return path
        except FileExistsError:
            continue


def pick_encoder(ffmpeg):
    """The H.264 encoder to use: the NVIDIA GPU encoder when it works here (many times faster), else libx264."""
    forced = os.environ.get("TIME_CUBE_ENCODER")
    if forced:
        return forced
    if not ffmpeg:
        return None
    try:
        r = subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=256x256:rate=30:duration=0.2",
                            "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True, timeout=20,
                           creationflags=NO_WINDOW)
        if r.returncode == 0:
            return "h264_nvenc"
    except Exception:
        pass
    return "libx264"


def video_codec_args(encoder, crf):
    """Codec options for a quality level (crf: 14 best .. 18 high, x264 scale)."""
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", str(int(crf) + 1),
                "-b:v", "0", "-maxrate", "120M", "-bufsize", "240M", "-spatial-aq", "1", "-profile:v", "high"]
    return ["-c:v", "libx264", "-preset", "fast", "-crf", str(int(crf))]


def even(n):
    return max(2, int(round(n / 2.0)) * 2)


def probe_media(ffprobe, path):
    """Duration, frame rate, display size (rotation applied) and whether there is sound."""
    cmd = [ffprobe, "-v", "error", "-show_entries",
           "format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,duration:"
           "stream_side_data=rotation:stream_tags=rotate", "-of", "json", path]
    r = subprocess.run(cmd, capture_output=True, text=True, creationflags=NO_WINDOW)
    if r.returncode != 0:
        raise ValueError("this file can't be read as a video")
    data = json.loads(r.stdout or "{}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video or not video.get("width"):
        raise ValueError("no video picture in this file")

    def rate(s):
        try:
            a, b = str(s).split("/")
            return float(a) / float(b) if float(b) else 0.0
        except Exception:
            return 0.0

    fps = rate(video.get("avg_frame_rate")) or rate(video.get("r_frame_rate")) or 30.0
    w, h = int(video["width"]), int(video["height"])
    rot = 0
    for sd in video.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    rot = rot or int(float((video.get("tags") or {}).get("rotate", 0) or 0))
    if abs(rot) % 180 == 90:
        w, h = h, w
    dur = float((data.get("format") or {}).get("duration") or video.get("duration") or 0)
    return {"width": w, "height": h, "fps": round(fps, 3), "duration": round(dur, 3),
            "audio": any(s.get("codec_type") == "audio" for s in streams), "codec": video.get("codec_name") or ""}


def clean_media(days=MEDIA_KEEP_DAYS):
    """Remove copies of opened videos, AI depth unused for that long, and logs older than `days`."""
    n, cutoff = 0, time.time() - days * 86400
    for folder in (MEDIA_DIR, DEPTH_DIR, os.path.join(DATA_DIR, "logs")):
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            p = os.path.join(folder, name)
            try:
                if os.path.isfile(p) and os.path.getmtime(p) < cutoff:
                    os.remove(p)
                    n += 1
            except OSError:
                pass
    return n


# ---- export sessions ---------------------------------------------------------------------------------------
class Session:
    """One ffmpeg process with a pipe, used by one export."""

    def __init__(self, cmd, kind, **pipes):
        os.makedirs(os.path.join(DATA_DIR, "logs"), exist_ok=True)
        self.log_path = os.path.join(DATA_DIR, "logs", f"{kind}-{secrets.token_hex(4)}.log")
        self.log = open(self.log_path, "wb")
        self.proc = subprocess.Popen(cmd, stderr=self.log, creationflags=NO_WINDOW, **pipes)
        no_throttle(self.proc.pid)
        self.lock = threading.Lock()
        self.touched = time.time()

    def error_text(self):
        try:
            self.log.flush()
            with open(self.log_path, "rb") as fh:
                return fh.read().decode("utf-8", "replace").strip()[-600:] or "ffmpeg stopped"
        except OSError:
            return "ffmpeg stopped"

    def kill(self):
        try:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait(5)
        except Exception:
            pass
        self.close_log()

    def close_log(self, remove=True):
        try:
            self.log.close()
            if remove:
                os.remove(self.log_path)
        except Exception:
            pass


class Decoder(Session):
    """Reads a source video as raw RGBA frames at a fixed frame rate and size."""

    def __init__(self, ffmpeg, src, fps, w, h, start, dur):
        self.frame_bytes = w * h * 4
        self.width, self.height = w, h
        self.done = False
        self.pos = 0          # frames handed out so far
        self.last = None      # (start, data, count, done) of the last batch, for a retried request
        cmd = [ffmpeg, "-v", "error", "-nostdin"]
        if start > 0:
            cmd += ["-ss", f"{start:.3f}"]
        cmd += ["-i", src]
        if dur > 0:
            cmd += ["-t", f"{dur:.3f}"]
        cmd += ["-an", "-sn", "-vf", f"fps={fps},scale={w}:{h}:flags=area,format=rgba",
                "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"]
        super().__init__(cmd, "decode", stdout=subprocess.PIPE, stdin=subprocess.DEVNULL)

    def read(self, n, start=None):
        """Up to n whole frames from frame `start` on, as bytes. Fewer (or none) at the end of the clip.

        Asking again for the same start returns the same batch, so a request lost on the way can be retried.
        """
        with self.lock:
            self.touched = time.time()
            if start is not None and self.last and start == self.last[0]:
                return self.last[1], self.last[2]
            if start is not None and start != self.pos:
                raise ValueError(f"asked for frame {start}, the next one is {self.pos}")
            if self.done:
                return b"", 0
            want = self.frame_bytes * n
            buf = bytearray()
            while len(buf) < want:
                c = self.proc.stdout.read(want - len(buf))
                if not c:
                    self.done = True
                    break
                buf += c
            whole = len(buf) // self.frame_bytes
            data = bytes(buf[:whole * self.frame_bytes])
            self.last = (self.pos, data, whole)
            self.pos += whole
            return data, whole


# Transparent video: the codec, the file type and the sound that goes with it. crf picks the quality level.
ALPHA_CODECS = {
    # ProRes 4444 (Best: 4444 XQ): the editing standard for alpha (Premiere, After Effects, Resolve, Final Cut)
    "prores": {"ext": "mov", "audio": ["-c:a", "pcm_s16le"],
               "video": lambda crf: ["-c:v", "prores_ks", "-profile:v", "5" if crf <= 14 else "4",
                                     "-pix_fmt", "yuva444p10le", "-vendor", "apl0"]},
    # VP9 with alpha in WebM: small files, plays with transparency in browsers
    "webm": {"ext": "webm", "audio": ["-c:a", "libopus", "-b:a", "160k"],
             "video": lambda crf: ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-crf", "16" if crf <= 14 else "22",
                                   "-b:v", "0", "-row-mt", "1", "-deadline", "good", "-cpu-used", "4"]},
}


class Encoder(Session):
    """Writes a video from what the window sends, plus the source video's sound.

    Kinds of input:
      raw    RGBA frames (bottom-up rows, straight from WebGL readPixels), encoded here with NVENC or libx264,
             or, with `alpha`, kept transparent as ProRes 4444 (.mov) or VP9 alpha (.webm);
      h264   an H.264 Annex B stream the window already encoded with WebCodecs; it is copied, not re-encoded.
    """

    def __init__(self, ffmpeg, w, h, fps, folder, stem, audio=None, flip=True, crf=16, encoder="libx264",
                 stream=None, alpha=None):
        self.stream = stream
        self.alpha = ALPHA_CODECS[alpha] if alpha else None
        self.frame_bytes = w * h * 4
        self.frames = 0       # raw: frames written
        self.parts = 0        # h264: chunks written
        if encoder == "h264_nvenc" and min(w, h) < 256:  # the GPU encoder refuses very small pictures
            encoder = "libx264"
        self.encoder = "copy" if stream else (alpha or encoder)
        ext = self.alpha["ext"] if self.alpha else "mp4"
        self.final = reserve(folder, stem, ext)
        self.tmp = self.final[:-len(ext) - 1] + f".part.{ext}"
        if stream == "h264":
            # -r on the input (not -framerate) times the frames exactly, whatever the stream's own timing says
            cmd = [ffmpeg, "-v", "error", "-y", "-f", "h264", "-r", str(fps), "-i", "pipe:0"]
        else:
            cmd = [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
                   "-r", str(fps), "-i", "pipe:0"]
        if audio:
            if audio.get("start", 0) > 0:
                cmd += ["-ss", f"{audio['start']:.3f}"]
            if audio.get("duration", 0) > 0:
                cmd += ["-t", f"{audio['duration']:.3f}"]
            cmd += ["-i", audio["path"], "-map", "0:v:0", "-map", "1:a:0"]
        if stream == "h264":
            cmd += ["-c:v", "copy", "-video_track_timescale", str(int(round(fps * 1000)))]
        else:
            if flip:
                cmd += ["-vf", "vflip"]
            if self.alpha:
                cmd += self.alpha["video"](int(crf)) + ["-r", str(fps)]
            else:
                cmd += video_codec_args(encoder, crf) + ["-pix_fmt", "yuv420p", "-r", str(fps)]
        if audio:
            cmd += self.alpha["audio"] if self.alpha else ["-c:a", "aac", "-b:a", "192k"]
        if ext != "webm":
            cmd += ["-movflags", "+faststart"]
        cmd += [self.tmp]
        try:
            super().__init__(cmd, "encode", stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
        except Exception:
            os.remove(self.final)
            raise

    def write(self, data, index=None):
        """Append data. `index` numbers the request (raw: its first frame; h264: the chunk), so a resent request
        is ignored and a missing one is noticed."""
        if not data or (not self.stream and len(data) % self.frame_bytes):
            raise ValueError("frame data has the wrong size")
        with self.lock:
            self.touched = time.time()
            done = self.parts if self.stream else self.frames
            if index is not None and index < done:
                return done
            if index is not None and index > done:
                raise ValueError(f"got part {index}, expected part {done}")
            try:
                self.proc.stdin.write(data)
            except (BrokenPipeError, OSError):
                raise RuntimeError("the video encoder stopped: " + self.error_text())
            if self.stream:
                self.parts += 1
                return self.parts
            self.frames += len(data) // self.frame_bytes
            return self.frames

    def finish(self):
        with self.lock:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            code = self.proc.wait()
            if code != 0 or not os.path.exists(self.tmp):
                msg = self.error_text()
                self.close_log(remove=False)
                self.discard()
                raise RuntimeError("the video encoder failed: " + msg)
            self.close_log()
            os.replace(self.tmp, self.final)
            return self.final

    def discard(self):
        self.kill()
        for p in (self.tmp, self.final):
            try:
                os.remove(p)
            except OSError:
                pass


# ---- AI depth ----------------------------------------------------------------------------------------------
def python_exe():
    """The Python that runs depth.py: this one, with the console python.exe in place of pythonw.exe."""
    env = os.environ.get("TIME_CUBE_PYTHON")
    if env:
        return env
    exe = sys.executable
    alt = os.path.join(os.path.dirname(exe), "python.exe")
    if os.path.basename(exe).lower() == "pythonw.exe" and os.path.exists(alt):
        return alt
    return exe


def depth_plan(info):
    """(fps, frames) of a video's AI depth: its own frame rate up to 30 fps, lower for very long videos."""
    fps = min(DEPTH_MAX_FPS, float(info.get("fps") or DEPTH_MAX_FPS))
    dur = float(info.get("duration") or 0)
    if dur * fps > DEPTH_MAX_FRAMES:
        fps = DEPTH_MAX_FRAMES / dur
    fps = max(1.0, round(fps, 3))
    return fps, max(1, int(round(dur * fps)))


def depth_base(media):
    """Where a video's depth lives, without the extension: named by the video's contents, version and frame rate."""
    fps, _ = depth_plan(media["info"])
    return os.path.join(DEPTH_DIR, f"{media['hash'][:24]}-v{DEPTH_VERSION}-{fps:g}fps")


def depth_meta(base):
    """The finished depth's description (depth.py writes the .json last), or None."""
    try:
        with open(base + ".json", encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta.get("version") == DEPTH_VERSION and os.path.exists(base + ".bin"):
            return meta
    except (OSError, ValueError):
        pass
    return None


class DepthJob:
    """One run of depth.py for one video, in its own process. Progress comes back as JSON lines."""

    def __init__(self, media, base):
        os.makedirs(DEPTH_DIR, exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, "logs"), exist_ok=True)
        self.base = base
        self.fps, self.total = depth_plan(media["info"])
        self.state = "running"      # running | ready | error | cancelled
        self.stage = "starting"
        self.done = 0
        self.ms = None
        self.error = None
        self.started = time.time()
        self.log_path = os.path.join(DATA_DIR, "logs", f"depth-{secrets.token_hex(4)}.log")
        self.log = open(self.log_path, "wb")
        cmd = [python_exe(), "-u", os.path.join(APP_DIR, "depth.py"), "--src", media["path"], "--out", base,
               "--fps", f"{self.fps:g}", "--long", str(DEPTH_LONG), "--parent", str(os.getpid())]
        # depth.py stops by itself when this server goes away (--parent)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=self.log,
                                     creationflags=NO_WINDOW)
        no_throttle(self.proc.pid)
        threading.Thread(target=self._follow, daemon=True).start()

    def _follow(self):
        for line in self.proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            kind = msg.get("t")
            if kind == "progress":
                self.done = int(msg.get("done") or 0)
                self.total = max(int(msg.get("total") or 0), self.done, 1)
                self.ms = msg.get("ms")
                self.stage = msg.get("stage") or "depth"
            elif kind == "error":
                self.error = msg.get("error")
        code = self.proc.wait()
        try:
            self.log.close()
        except OSError:
            pass
        if self.state == "cancelled":
            return
        if code == 0 and depth_meta(self.base):
            self.state = "ready"
            try:
                os.remove(self.log_path)
            except OSError:
                pass
            return
        if not self.error:
            try:
                with open(self.log_path, "rb") as fh:
                    tail = fh.read().decode("utf-8", "replace").strip()[-300:]
            except OSError:
                tail = ""
            self.error = f"the depth process stopped ({tail or f'exit code {code}'})"
        self.state = "error"

    def cancel(self):
        if self.state == "running":
            self.state = "cancelled"
        try:
            if self.proc.poll() is None:
                self.proc.kill()
        except Exception:
            pass

    def status(self):
        return {"state": self.state, "stage": self.stage, "done": self.done, "total": self.total, "ms": self.ms,
                "error": self.error, "fps": self.fps, "seconds": round(time.time() - self.started, 1)}


def depth_status(media):
    """Where a video's AI depth stands: none | running | ready | error | cancelled, with progress or its size."""
    base = depth_base(media)
    meta = depth_meta(base)
    if meta:
        return {"state": "ready", "fps": meta["fps"], "frames": meta["frames"], "width": meta["width"],
                "height": meta["height"], "msPerFrame": meta.get("msPerFrame")}
    with State.lock:
        job = State.depth.get(base)
    fps, frames = depth_plan(media["info"])
    # a first guess: loading the model, then about 20 ms a frame on this PC's graphics card
    plan = {"state": "none", "fps": fps, "frames": frames, "estimate": round(6 + frames * 0.022)}
    if job:
        return {**plan, **job.status()}
    return plan


def depth_start(media):
    base = depth_base(media)
    if depth_meta(base):
        return depth_status(media)
    with State.lock:
        job = State.depth.get(base)
        if job and job.state == "running":
            return job.status()
        # one depth run at a time: the graphics card is shared with the preview
        others = [j for j in State.depth.values() if j.state == "running"]
    for j in others:
        j.cancel()
    job = DepthJob(media, base)
    with State.lock:
        State.depth[base] = job
    return job.status()


_path_locks = {}


def depth_path(media, fov):
    """The video's camera path for a lens (measured now if it isn't yet: about a second for 5 s of video)."""
    base = depth_base(media)
    if not depth_meta(base):
        raise ValueError("this video has no AI depth yet: build it first")
    fov = int(round(max(20.0, min(120.0, float(fov)))))
    dst = f"{base}.path-fov{fov}.json"
    with State.lock:
        lock = _path_locks.setdefault(dst, threading.Lock())
    with lock:                          # two asks at once measure once
        if not os.path.exists(dst):
            cmd = [python_exe(), "-u", os.path.join(APP_DIR, "path.py"), "--src", media["path"], "--depth", base,
                   "--fov", str(fov), "--parent", str(os.getpid())]
            r = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=PATH_TIMEOUT,
                               creationflags=NO_WINDOW)
            if r.returncode != 0 or not os.path.exists(dst):
                msg = next((json.loads(ln).get("error") for ln in r.stdout.splitlines() if ln.startswith("{")), None)
                raise RuntimeError(f"the camera path couldn't be measured: {msg or r.stderr.strip()[-300:]}")
        now = time.time()
        os.utime(dst, (now, now))
        with open(dst, "rb") as fh:
            return fh.read()


def depth_cancel(media):
    with State.lock:
        job = State.depth.get(depth_base(media))
    if job:
        job.cancel()
    return depth_status(media)


# ---- object tracking ---------------------------------------------------------------------------------------
def mask_size(media):
    """A track's masks: the size of the video's AI depth (made or not), so the two line up."""
    meta = depth_meta(depth_base(media))
    if meta:
        return meta["width"], meta["height"]
    w, h = media["info"]["width"], media["info"]["height"]
    a = w / max(1, h)
    return (even(DEPTH_LONG), even(DEPTH_LONG / a)) if a >= 1 else (even(DEPTH_LONG * a), even(DEPTH_LONG))


def track_grid(media):
    """(fps, frames) of the video's frames: its depth's, or the plan the depth would have."""
    meta = depth_meta(depth_base(media))
    return (meta["fps"], meta["frames"]) if meta else depth_plan(media["info"])


class Worker:
    """A GPU helper (track.py, caption.py) run as `<script> --serve`: started when first needed, one command at a
    time, closed after TRACK_IDLE idle seconds."""

    def __init__(self, script, python=None):
        self.script = script
        self.python = python or python_exe
        self.proc = None
        self.lines = None
        self.lock = threading.Lock()
        self.touched = time.time()
        self.busy = False
        self.log_path = None

    def ensure(self):
        if self.proc and self.proc.poll() is None:
            return
        os.makedirs(os.path.join(DATA_DIR, "logs"), exist_ok=True)
        name = os.path.splitext(self.script)[0]
        self.log_path = os.path.join(DATA_DIR, "logs", f"{name}-{secrets.token_hex(4)}.log")
        log = open(self.log_path, "wb")
        try:
            self.proc = subprocess.Popen([self.python(), "-u", os.path.join(APP_DIR, self.script), "--serve", "--parent",
                                          str(os.getpid())], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                         creationflags=NO_WINDOW)
        finally:
            log.close()          # the process has its own handle
        no_throttle(self.proc.pid)
        lines = self.lines = queue.Queue()

        def read(proc):
            for raw in proc.stdout:
                try:
                    lines.put(json.loads(raw))
                except ValueError:
                    pass
            lines.put(None)
        threading.Thread(target=read, args=(self.proc,), daemon=True).start()

    def ask(self, cmd, wait=240, progress=None):
        """Send one command and wait for its answer (progress lines go to `progress`). `wait`: seconds between
        lines before giving up (loading the models the first time takes a while)."""
        with self.lock:
            self.busy = True
            try:
                self.ensure()
                self.proc.stdin.write((json.dumps(cmd) + "\n").encode("utf-8"))
                self.proc.stdin.flush()
                while True:
                    try:
                        msg = self.lines.get(timeout=wait)
                    except queue.Empty:
                        self.kill()
                        raise RuntimeError(f"the {self.script} process stopped answering")
                    if msg is None:
                        raise RuntimeError(f"the {self.script} process stopped" + self.log_tail())
                    if msg.get("t") == "progress":
                        if progress:
                            progress(msg)
                        continue
                    if msg.get("t") == "error":
                        raise RuntimeError(msg.get("error") or f"{self.script} failed")
                    return msg
            except (BrokenPipeError, OSError) as e:
                raise RuntimeError(f"the {self.script} process stopped ({e})")
            finally:
                self.busy = False
                self.touched = time.time()

    def log_tail(self):
        try:
            with open(self.log_path, "rb") as fh:
                t = fh.read().decode("utf-8", "replace").strip()
            return f": {t[-300:]}" if t else ""
        except (OSError, TypeError):
            return ""

    def kill(self):
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.kill()
        except Exception:
            pass

    def idle(self):
        """Close the worker when nobody has used it for a while."""
        if self.proc and self.proc.poll() is None and not self.busy and time.time() - self.touched > TRACK_IDLE:
            self.kill()


def caption_python():
    exe = os.environ.get("TIME_CUBE_CAPTION_PYTHON") or os.path.join(CAPTION_ENV, "Scripts", "python.exe")
    if not os.path.exists(exe):
        raise ValueError("captions need their small Python environment: run make_caption_env.ps1 in the app folder")
    return exe


TRACKER = Worker("track.py")
CAPTIONER = Worker("caption.py", python=caption_python)


def track_base(media):
    return depth_base(media) + ".track"


def track_meta(base):
    try:
        with open(base + ".json", encoding="utf-8") as fh:
            meta = json.load(fh)
        if os.path.exists(base + ".bin"):
            return meta
    except (OSError, ValueError):
        pass
    return None


def frame_of(media, t):
    fps, frames = track_grid(media)
    return fps, frames, max(0, min(frames - 1, int(round(float(t or 0) * fps))))


def track_busy():
    with State.lock:
        return any(f["state"] == "running" for f in State.follows.values())


def track_ask(media, cmd, **args):
    if track_busy():
        raise ValueError("an object is being followed: wait for it, or press Stop")
    fps, frames, k = frame_of(media, args.pop("time", 0))
    return TRACKER.ask({"cmd": cmd, "src": media["path"], "fps": fps, "frame": k, **args})


def track_follow(media, d):
    base = track_base(media)
    fps, frames, k = frame_of(media, d.get("time"))
    with State.lock:
        if any(f["state"] == "running" for f in State.follows.values()):
            raise ValueError("an object is already being followed")
        job = State.follows[base] = {"state": "running", "done": 0, "total": frames, "label": str(d.get("label") or "")[:60],
                                     "error": None, "started": time.time()}
    dbase = depth_base(media)
    cmd = {"cmd": "follow", "src": media["path"], "fps": fps, "frames": frames, "frame": k, "box": d.get("box"),
           "point": d.get("point"), "outline": d.get("outline"), "label": job["label"], "out": base,
           "size": list(mask_size(media)),
           "depth": dbase + ".bin" if depth_meta(dbase) else None}

    def run():
        def progress(msg):
            job["done"] = int(msg.get("done") or 0)
        try:
            TRACKER.ask(cmd, wait=180, progress=progress)
            if job["state"] == "running":
                job["state"] = "ready"
        except Exception as e:  # noqa: BLE001
            if job["state"] == "running":
                job["state"], job["error"] = "error", str(e)
    threading.Thread(target=run, daemon=True).start()
    return track_status(media)


def track_status(media):
    base = track_base(media)
    with State.lock:
        job = dict(State.follows.get(base) or {})
    meta = track_meta(base)
    if job.get("state") == "running":
        return {**job, "seconds": round(time.time() - job["started"], 1)}
    if meta:
        return {"state": "ready", "label": meta.get("label"), "frames": meta["frames"], "seen": meta["seen"],
                "fps": meta["fps"], "width": meta["width"], "height": meta["height"], "seconds": meta.get("seconds")}
    if job:
        return job
    fps, frames = track_grid(media)
    return {"state": "none", "frames": frames, "fps": fps}


def track_cancel(media):
    base = track_base(media)
    with State.lock:
        job = State.follows.get(base)
    if job and job["state"] == "running":
        job["state"] = "cancelled"
        TRACKER.kill()            # the quickest way to stop it; it starts again when needed
    return track_status(media)


def track_clear(media):
    base = track_base(media)
    track_cancel(media)
    for ext in (".bin", ".json"):
        try:
            os.remove(base + ext)
        except OSError:
            pass
    with State.lock:
        State.follows.pop(base, None)
    return track_status(media)


# ---- what the AI sees ---------------------------------------------------------------------------------------
def caption_base(media):
    return depth_base(media) + ".captions"


def caption_meta(base):
    try:
        with open(base + ".json", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def caption_times(duration):
    """The moments described: 3 to 8 of them, spread over the video."""
    d = max(0.1, float(duration or 0))
    n = max(3, min(8, int(round(d / 2.5))))
    lo, hi = min(0.3, d * 0.05), max(min(0.3, d * 0.05), d - min(0.3, d * 0.05))
    return [round(lo + (hi - lo) * i / (n - 1), 2) for i in range(n)]


def caption_busy():
    with State.lock:
        return any(j["state"] == "running" for j in State.describes.values())


def caption_describe(media):
    base = caption_base(media)
    caption_python()      # a clear message now if the environment isn't there
    with State.lock:
        if any(j["state"] == "running" for j in State.describes.values()):
            raise ValueError("a video is being described: wait a moment")
        times = caption_times(media["info"]["duration"])
        job = State.describes[base] = {"state": "running", "done": 0, "total": len(times) + 1, "error": None,
                                       "started": time.time()}

    def run():
        try:
            r = CAPTIONER.ask({"cmd": "describe", "src": media["path"], "times": times}, wait=420,
                              progress=lambda m: job.update(done=int(m.get("done") or 0)))
            out = {"moments": r["moments"], "place": r.get("place", ""), "seconds": r.get("seconds"),
                   "model": "Qwen3-VL 4B"}
            with open(base + ".json.part", "w", encoding="utf-8") as fh:
                json.dump(out, fh)
            os.replace(base + ".json.part", base + ".json")
            if job["state"] == "running":
                job["state"] = "ready"
        except Exception as e:  # noqa: BLE001
            if job["state"] == "running":
                job["state"], job["error"] = "error", str(e)
    threading.Thread(target=run, daemon=True).start()
    return caption_status(media)


def caption_status(media):
    base = caption_base(media)
    with State.lock:
        job = dict(State.describes.get(base) or {})
    meta = caption_meta(base)
    if job.get("state") == "running":
        return {**job, "seconds": round(time.time() - job["started"], 1)}
    if meta:
        return {"state": "ready", **meta}
    return job or {"state": "none", "moments": len(caption_times(media["info"]["duration"]))}


def caption_ask(media, d):
    if caption_busy():
        raise ValueError("the video is being described: ask again in a moment")
    q = str(d.get("question") or "").strip()
    if not q:
        raise ValueError("type a question about this moment")
    caption_python()
    t = max(0.0, min(float(media["info"]["duration"] or 0), float(d.get("time") or 0)))
    return CAPTIONER.ask({"cmd": "ask", "src": media["path"], "time": t, "question": q[:400]}, wait=240)


def caption_clear(media):
    base = caption_base(media)
    try:
        os.remove(base + ".json")
    except OSError:
        pass
    with State.lock:
        State.describes.pop(base, None)
    return caption_status(media)


def depth_cancel_all():
    with State.lock:
        jobs = list(State.depth.values())
    for j in jobs:
        j.cancel()


class State:
    last_beat = time.time()
    beaten = False
    busy = 0
    lock = threading.Lock()
    media = {}       # id -> {"path", "name", "info"}
    sessions = {}    # id -> Decoder | Encoder
    depth = {}       # depth file base -> DepthJob
    follows = {}     # track file base -> {"state", "done", "total", "label", "error", "started"}
    describes = {}   # captions file base -> {"state", "done", "total", "error", "started"}


def new_session(sess):
    sid = secrets.token_hex(8)
    with State.lock:
        State.sessions[sid] = sess
    return sid


def get_session(sid, kind):
    with State.lock:
        s = State.sessions.get(sid)
    if not isinstance(s, kind):
        raise ValueError("this export session has ended")
    return s


def drop_session(sid):
    with State.lock:
        return State.sessions.pop(sid, None)


def reap_sessions():
    """Close export sessions the window has abandoned (closed mid-export, crashed...)."""
    now = time.time()
    with State.lock:
        stale = [k for k, s in State.sessions.items() if now - s.touched > SESSION_IDLE]
    for k in stale:
        s = drop_session(k)
        if isinstance(s, Encoder):
            s.discard()
        elif s:
            s.kill()


class Handler(BaseHTTPRequestHandler):
    server_version = "TimeCube/1.0"
    protocol_version = "HTTP/1.1"  # keep-alive: an export sends thousands of requests

    def log_message(self, fmt, *args):  # keep the console and log quiet
        pass

    def log_failure(self, e):
        print(f"{time.strftime('%H:%M:%S')} {self.command} {self.path}: {e!r}", flush=True)
        if not isinstance(e, ValueError):
            traceback.print_exc()

    def drain(self):
        """Read what is left of the request body, so an early error reply doesn't reset the connection."""
        left = int(self.headers.get("Content-Length") or 0) - getattr(self, "_consumed", 0)
        try:
            while left > 0:
                c = self.rfile.read(min(left, 1 << 22))
                if not c:
                    break
                left -= len(c)
                self._consumed = getattr(self, "_consumed", 0) + len(c)
        except Exception:
            self.close_connection = True

    # ---- helpers -------------------------------------------------------------------------------------------
    def send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(self, body, headers=None):
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, str(v))
        self.end_headers()
        self.wfile.write(body)

    def body_length(self, limit):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            raise ValueError("nothing was sent")
        if n > limit:
            raise ValueError("too large")
        return n

    def read_body(self, limit=MAX_FRAME * 4):
        left = self.body_length(limit)
        chunks = []
        while left > 0:
            c = self.rfile.read(min(left, 1 << 22))
            if not c:
                raise ValueError("connection closed during upload")
            chunks.append(c)
            left -= len(c)
            self._consumed = getattr(self, "_consumed", 0) + len(c)
        return b"".join(chunks)

    def read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        if n > 1 << 16:
            raise ValueError("request too large")
        data = self.rfile.read(n)
        self._consumed = getattr(self, "_consumed", 0) + len(data)
        return json.loads(data or b"{}")

    def need(self, tool):
        exe = getattr(self.server, tool)
        if not exe:
            raise ValueError(f"{tool} was not found on this computer")
        return exe

    # ---- routes --------------------------------------------------------------------------------------------
    def do_GET(self):
        self._consumed = 0
        url = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/api/ping":
                return self.send_json({"app": APP_ID, "ffmpeg": bool(self.server.ffmpeg), "outDir": output_dir(),
                                   "encoder": self.server.encoder})
            if url.path == "/api/depth/status":
                return self.send_json(depth_status(self.media(q.get("id"))))
            if url.path == "/api/depth/data":
                return self.depth_data(q)
            if url.path == "/api/caption/status":
                return self.send_json(caption_status(self.media(q.get("id"))))
            if url.path == "/api/track/status":
                return self.send_json(track_status(self.media(q.get("id"))))
            if url.path in ("/api/track/data", "/api/track/info"):
                return self.track_data(q, url.path.endswith("info"))
            if url.path == "/api/depth/path":
                body = depth_path(self.media(q.get("id")), q.get("fov") or 65)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            m = re.fullmatch(r"/api/decode/([0-9a-f]+)/frames", url.path)
            if m:
                dec = get_session(m.group(1), Decoder)
                n = max(1, min(64, int(q.get("n") or 1)))
                data, count = dec.read(n, int(q["from"]) if q.get("from") else None)
                return self.send_bytes(data, {"X-Frames": count, "X-Done": int(dec.done)})
        except (ConnectionError, TimeoutError) as e:
            self.log_failure(e)
            self.close_connection = True
            return
        except ValueError as e:
            self.log_failure(e)
            return self.send_json({"error": str(e)}, 400)
        except Exception as e:  # noqa: BLE001 - report every failure to the window
            self.log_failure(e)
            return self.send_json({"error": str(e)}, 500)
        return self.static(url.path)

    def static(self, path):
        rel = unquote(path).lstrip("/") or "index.html"
        full = os.path.normpath(os.path.join(STATIC, rel))
        if not full.startswith(STATIC + os.sep) or not os.path.isfile(full):
            return self.send_error(HTTPStatus.NOT_FOUND)
        with open(full, "rb") as fh:
            body = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(full)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self._consumed = 0  # this handler serves every request of a kept-alive connection
        url = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(url.query).items()}
        with State.lock:
            State.busy += 1
        try:
            p = url.path
            if p == "/api/heartbeat":
                State.last_beat, State.beaten = time.time(), True
                self.drain()
                return self.send_json({"ok": True})
            if p == "/api/upload":
                return self.upload(q)
            if p == "/api/save":
                return self.save(q)
            if p == "/api/reveal":
                return self.reveal()
            if p == "/api/depth/start":
                return self.send_json(depth_start(self.media(self.read_json().get("id"))))
            if p.startswith("/api/track/"):
                return self.track_call(p.rsplit("/", 1)[-1])
            if p.startswith("/api/caption/"):
                d = self.read_json()
                m = self.media(d.get("id"))
                action = p.rsplit("/", 1)[-1]
                if action == "describe":
                    return self.send_json(caption_describe(m))
                if action == "ask":
                    return self.send_json(caption_ask(m, d))
                if action == "clear":
                    return self.send_json(caption_clear(m))
                raise ValueError("unknown request")
            if p == "/api/depth/cancel":
                return self.send_json(depth_cancel(self.media(self.read_json().get("id"))))
            if p == "/api/decode/open":
                return self.decode_open()
            if p == "/api/encode/open":
                return self.encode_open()
            m = re.fullmatch(r"/api/(decode|encode)/([0-9a-f]+)/(frame|finish|close)", p)
            if m:
                return self.session_call(*m.groups(), q)
            self.drain()
            return self.send_json({"error": "not found"}, 404)
        except (ConnectionError, TimeoutError) as e:  # the window went away mid-request
            self.log_failure(e)
            self.close_connection = True
        except ValueError as e:
            self.log_failure(e)
            self.drain()
            return self.send_json({"error": str(e)}, 400)
        except Exception as e:  # noqa: BLE001
            self.log_failure(e)
            self.drain()
            return self.send_json({"error": str(e)}, 500)
        finally:
            self.drain()  # never leave body bytes on a kept-alive connection
            with State.lock:
                State.busy -= 1

    def upload(self, q):
        """The window sends a copy of the video it opened; it is streamed to disk and probed."""
        ffprobe = self.need("ffprobe")
        left = self.body_length(MAX_UPLOAD)
        name = os.path.basename(str(q.get("name") or "video.mp4").replace("\\", "/"))
        ext = os.path.splitext(name)[1].lower()
        ext = ext if re.fullmatch(r"\.[a-z0-9]{1,5}", ext or "") else ".mp4"
        os.makedirs(MEDIA_DIR, exist_ok=True)
        mid = secrets.token_hex(8)
        path = os.path.join(MEDIA_DIR, mid + ext)
        digest = hashlib.sha1()   # names the video's AI depth, so the same video finds it again
        try:
            with open(path, "wb") as fh:
                while left > 0:
                    c = self.rfile.read(min(left, 1 << 22))
                    if not c:
                        raise ValueError("connection closed during upload")
                    fh.write(c)
                    digest.update(c)
                    left -= len(c)
                    self._consumed = getattr(self, "_consumed", 0) + len(c)
            info = probe_media(ffprobe, path)
        except Exception:
            try:
                os.remove(path)
            except OSError:
                pass
            raise
        with State.lock:
            State.media[mid] = {"path": path, "name": name, "info": info, "hash": digest.hexdigest()}
        return self.send_json({"id": mid, "name": name, **info})

    def media(self, mid):
        with State.lock:
            m = State.media.get(str(mid))
        if not m or not os.path.exists(m["path"]):
            raise ValueError("open the video again: its copy is gone")
        return m

    def track_call(self, action):
        d = self.read_json()
        m = self.media(d.get("id"))
        if action == "find":
            text = str(d.get("text") or "").strip()
            if not text:
                raise ValueError("type what you're looking for")
            return self.send_json(track_ask(m, "find", time=d.get("time"), text=text))
        if action == "here":
            return self.send_json(track_ask(m, "here", time=d.get("time")))
        if action == "cut":
            pt = d.get("point")
            if not (isinstance(pt, list) and len(pt) == 2):
                raise ValueError("click on the thing to cut it out")
            return self.send_json(track_ask(m, "cut", time=d.get("time"), point=[float(pt[0]), float(pt[1])]))
        if action == "follow":
            if not d.get("box") and not d.get("point"):
                raise ValueError("pick something to follow first")
            return self.send_json(track_follow(m, d))
        if action == "cancel":
            return self.send_json(track_cancel(m))
        if action == "clear":
            return self.send_json(track_clear(m))
        raise ValueError("unknown request")

    def track_data(self, q, info):
        """The followed object: its masks (bytes, a frame after the other) or, with info, every frame's centre."""
        m = self.media(q.get("id"))
        base = track_base(m)
        meta = track_meta(base)
        if not meta:
            raise ValueError("nothing has been followed in this video yet")
        now = time.time()
        for f in (base + ".bin", base + ".json"):
            os.utime(f, (now, now))
        if info:
            return self.send_json(meta)
        path = base + ".bin"
        size = os.path.getsize(path)
        if size != meta["frames"] * meta["width"] * meta["height"]:
            raise ValueError("the saved track is damaged: follow it again")
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        for k, v in (("X-Width", meta["width"]), ("X-Height", meta["height"]), ("X-Frames", meta["frames"]),
                     ("X-Fps", meta["fps"])):
            self.send_header(k, str(v))
        self.end_headers()
        with open(path, "rb") as fh:
            shutil.copyfileobj(fh, self.wfile, 1 << 20)

    def depth_data(self, q):
        """A video's finished AI depth: every frame's bytes one after the other (255 = nearest)."""
        m = self.media(q.get("id"))
        base = depth_base(m)
        meta = depth_meta(base)
        if not meta:
            raise ValueError("this video has no AI depth yet")
        path = base + ".bin"
        size = os.path.getsize(path)
        if size != meta["frames"] * meta["width"] * meta["height"]:
            raise ValueError("the saved depth is damaged: build it again")
        now = time.time()
        for f in (path, base + ".json"):   # in use: kept for another few days
            os.utime(f, (now, now))
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        for k, v in (("X-Width", meta["width"]), ("X-Height", meta["height"]), ("X-Frames", meta["frames"]),
                     ("X-Fps", meta["fps"])):
            self.send_header(k, str(v))
        self.end_headers()
        with open(path, "rb") as fh:
            shutil.copyfileobj(fh, self.wfile, 1 << 20)

    def decode_open(self):
        d = self.read_json()
        m = self.media(d.get("id"))
        fps = float(d.get("fps") or 30)
        w, h = even(float(d.get("width") or 0)), even(float(d.get("height") or 0))
        if not (1 <= fps <= 120) or w * h * 4 > MAX_FRAME:
            raise ValueError("bad decode size or frame rate")
        start = max(0.0, float(d.get("start") or 0))
        dur = max(0.0, float(d.get("duration") or 0))
        dec = Decoder(self.need("ffmpeg"), m["path"], fps, w, h, start, dur)
        total = m["info"]["duration"] - start if dur <= 0 else dur
        return self.send_json({"sid": new_session(dec), "width": w, "height": h,
                               "frames": max(1, int(round(max(0.0, total) * fps)))})

    def encode_open(self):
        d = self.read_json()
        w, h, fps = int(d.get("width") or 0), int(d.get("height") or 0), float(d.get("fps") or 30)
        if w < 16 or h < 16 or w % 2 or h % 2 or w * h * 4 > MAX_FRAME or not (1 <= fps <= 120):
            raise ValueError("bad video size or frame rate")
        audio = None
        a = d.get("audio")
        if a and a.get("id"):
            m = self.media(a["id"])
            if m["info"].get("audio"):
                audio = {"path": m["path"], "start": float(a.get("start") or 0), "duration": float(a.get("duration") or 0)}
        folder = output_dir()
        if d.get("folder"):
            folder = os.path.join(folder, safe_stem(d["folder"]))
        stream = d.get("stream")
        if stream not in (None, "h264"):
            raise ValueError("unknown stream type")
        alpha = d.get("alpha")
        if alpha not in (None, *ALPHA_CODECS):
            raise ValueError("unknown transparent video type")
        if alpha and stream:
            raise ValueError("a transparent video needs raw frames")
        enc = Encoder(self.need("ffmpeg"), w, h, fps, folder, safe_stem(d.get("name")), audio,
                      flip=d.get("flip", True), crf=int(d.get("crf") or 16), encoder=self.server.encoder,
                      stream=stream, alpha=alpha)
        return self.send_json({"sid": new_session(enc), "path": enc.final, "audio": bool(audio),
                               "encoder": enc.encoder})

    def session_call(self, kind, sid, action, q=None):
        cls = Decoder if kind == "decode" else Encoder
        if action == "close":
            s = drop_session(sid)
            if isinstance(s, Encoder):
                s.discard()
            elif s:
                s.kill()
            return self.send_json({"ok": True})
        s = get_session(sid, cls)
        if kind == "encode" and action == "frame":
            index = int(q["i"]) if q and q.get("i") else None
            return self.send_json({"frames": s.write(self.read_body(), index)})
        if kind == "encode" and action == "finish":
            drop_session(sid)
            path = s.finish()
            frames = s.frames if not s.stream else int((q or {}).get("frames") or 0)
            return self.send_json({"path": path, "frames": frames, "bytes": os.path.getsize(path),
                                   "encoder": s.encoder})
        raise ValueError("unknown request")

    def save(self, q):
        """A PNG still, or a live recording (WebM or MP4 from MediaRecorder) that ffmpeg turns into a clean MP4."""
        ext = (q.get("ext") or "").lower().lstrip(".")
        if ext not in {"png", "mp4"}:
            raise ValueError(f"unsupported file type: {ext}")
        data = self.read_body(MAX_UPLOAD if ext == "mp4" else MAX_FRAME)
        folder = output_dir()
        if q.get("folder"):
            folder = os.path.join(folder, safe_stem(q["folder"]))
        path = reserve(folder, safe_stem(q.get("name")), ext)
        if ext == "png":
            with open(path, "wb") as fh:
                fh.write(data)
        else:
            tmp = path[:-4] + ".part.webm"
            with open(tmp, "wb") as fh:
                fh.write(data)
            path = self.transcode_recording(tmp, path, q)
        return self.send_json({"path": path, "bytes": os.path.getsize(path)})

    def transcode_recording(self, tmp, path, q):
        """A live WebM recording -> MP4 at one fixed size and a constant frame rate.

        The picture is scaled to the size the window recorded at (so a size change at the very start can't stop
        the encoder). If the GPU encoder fails, libx264 tries; if that fails too, the WebM itself is kept, so a
        recording is never lost. Returns the path that was saved.
        """
        w, h = even(float(q.get("w") or 0)), even(float(q.get("h") or 0))
        fps = max(1.0, min(120.0, float(q.get("fps") or 30)))
        size = f"scale={w}:{h}:flags=bicubic" if w > 2 and h > 2 else "scale=trunc(iw/2)*2:trunc(ih/2)*2"
        err = ""
        part = path[:-4] + ".part.mp4"   # the final name only appears once the file is complete
        for enc in dict.fromkeys([self.server.encoder, "libx264"]):
            cmd = [self.need("ffmpeg"), "-v", "error", "-y", "-i", tmp, "-vf", size, "-fps_mode", "cfr", "-r",
                   f"{fps:g}"] + video_codec_args(enc, 16) + ["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                                                              "-movflags", "+faststart", part]
            r = subprocess.run(cmd, capture_output=True, text=True, creationflags=NO_WINDOW)
            if r.returncode == 0:
                os.remove(tmp)
                os.replace(part, path)
                return path
            err = (r.stderr or "ffmpeg failed").strip()[-400:]
            print(f"recording transcode with {enc} failed: {err}", flush=True)
        for leftover in (part, path):
            try:
                os.remove(leftover)
            except OSError:
                pass
        keep = reserve(os.path.dirname(path), os.path.splitext(os.path.basename(path))[0], "webm")
        os.replace(tmp, keep)
        return keep

    def reveal(self):
        data = self.read_json()
        path = os.path.normpath(str(data.get("path", "")))
        out = os.path.normpath(output_dir())
        if not path.startswith(out + os.sep) or not os.path.exists(path):
            return self.send_json({"error": "not an export of this app"}, 400)
        subprocess.Popen(["explorer", "/select,", path])
        return self.send_json({"ok": True})


def make_server(port=PORT):
    os.makedirs(MEDIA_DIR, exist_ok=True)
    os.makedirs(DEPTH_DIR, exist_ok=True)
    clean_media()
    ThreadingHTTPServer.request_queue_size = 64
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    srv.ffmpeg = find_tool("ffmpeg")
    srv.ffprobe = find_tool("ffprobe")
    srv.encoder = pick_encoder(srv.ffmpeg)
    no_throttle()
    return srv


def serve(auto_exit=False, port=PORT):
    srv = make_server(port)

    def watchdog():
        while True:
            time.sleep(2)
            reap_sessions()
            TRACKER.idle()
            CAPTIONER.idle()
            if not auto_exit:
                continue
            idle = time.time() - State.last_beat
            if (State.beaten and idle > IDLE_EXIT or not State.beaten and idle > 120) and State.busy == 0 \
                    and not State.sessions:
                srv.shutdown()
                return

    threading.Thread(target=watchdog, daemon=True).start()
    print(f"{APP_NAME} on http://127.0.0.1:{port}/  ->  saving to {output_dir()}", flush=True)
    try:
        srv.serve_forever()
    finally:
        depth_cancel_all()
        TRACKER.kill()
        CAPTIONER.kill()


if __name__ == "__main__":
    if sys.stdout is None or sys.stderr is None:
        os.makedirs(DATA_DIR, exist_ok=True)
        sys.stdout = sys.stderr = open(os.path.join(DATA_DIR, "app.log"), "a", buffering=1, encoding="utf-8")
    serve(auto_exit="--auto-exit" in sys.argv)
