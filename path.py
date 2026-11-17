"""Camera path for Time Cube: where the camera was for every frame, measured from the video and its AI depth.

    python path.py --src <video> --depth <depth file, without extension> [--fov 65] [--parent <pid>]

Points are found in a frame and followed through the next frames (OpenCV optical flow, checked backwards). When the
picture has moved enough, the camera's move since that keyframe is solved from the points' 3D positions (the AI depth
of the keyframe) and where they ended up (PnP with RANSAC, so people walking through the shot don't count). Moves are
chained into a path, frames between keyframes are filled in, and the path is smoothed a little.

Depth Anything's depth has no unit, so the path is in the same made-up unit as the depth: a nearness n (0 far ..
1 near, the depth file's bytes / 255) is the distance 1 / (C + (1 - C) n), from 1 (nearest) to 1 / C (the far
distance). The window uses the same rule to place every frame's points, so the frames line up with their path.
The lens isn't stored in a video: fov is the assumed field of view across the picture's long side.

Writes <depth>.path-fov<fov>.json: every frame's camera-to-world matrix (OpenGL axes: x right, y up, the camera
looking down -z; column-major, the first frame is the identity) and how many keyframe steps were solved.
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

PATH_VERSION = 1
DEPTH_C = 0.08          # nearness -> distance: 1 / (C + (1 - C) n), so from 1 (nearest) to 12.5 (far)
TRACK_LONG = 640        # the frames the points are followed in: long side in pixels
MIN_FLOW = 12.0         # solve once the points have moved this far (pixels, median) ...
MAX_GAP = 10            # ... or after this many frames
STILL_FLOW = 0.6        # below this the camera counts as still
MIN_POINTS = 30
SMOOTH_SIGMA = 1.5      # frames
S = np.diag([1.0, -1.0, -1.0, 1.0])   # OpenCV camera axes (y down, z forward) <-> OpenGL (y up, z back)


def distance(nearness):
    """Nearness 0..1 -> distance (the path's unit)."""
    return 1.0 / (DEPTH_C + (1.0 - DEPTH_C) * nearness)


def intrinsics(w, h, fov):
    f = (max(w, h) / 2.0) / np.tan(np.radians(fov) / 2.0)
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], np.float64)


def _track_points(cv2, gray, near):
    """Corners worth following: not the sky or far haze, not on a depth edge (where the depth is unsure)."""
    gy, gx = np.gradient(near)
    mask = ((near > 0.04) & (np.hypot(gx, gy) < 0.02)).astype(np.uint8) * 255
    p = cv2.goodFeaturesToTrack(gray, 2000, 0.008, 6, mask=mask)
    return p if p is not None else np.zeros((0, 1, 2), np.float32)


def _smooth(values, sigma):
    """Gaussian smoothing along the first axis (the ends hold their value)."""
    if sigma <= 0 or len(values) < 3:
        return values
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.repeat(values[:1], r, 0), values, np.repeat(values[-1:], r, 0)])
    return np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(values.shape[1])], 1)


def measure(gray, nearness, fov, progress=None):
    """Camera-to-world poses (OpenCV axes) for every frame.

    gray: (N, h, w) uint8 frames; nearness: (N, hd, wd) float 0..1 (any size, it is scaled to the frames).
    Returns (poses (N, 4, 4), measured (N,) bool: a keyframe whose move was solved, steps tried, steps solved).
    """
    import cv2
    from scipy.spatial.transform import Rotation, Slerp

    n, h, w = gray.shape
    K = intrinsics(w, h, fov)
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    near_at = lambda k: cv2.resize(nearness[k].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)  # noqa: E731
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    keys = {0: np.eye(4)}          # keyframe -> camera-to-world
    measured = np.zeros(n, bool)
    measured[0] = True
    last_move, last_gap = np.eye(4), 1   # the last solved move, for filling a failed step
    tried = solved = 0
    key = 0
    near = near_at(0)
    kp = _track_points(cv2, gray[0], near)
    kz = distance(near)
    cur, alive = kp.copy(), np.ones(len(kp), bool)
    k = 0
    while k < n - 1:
        if len(cur):
            p1, st, _ = cv2.calcOpticalFlowPyrLK(gray[k], gray[k + 1], cur, None, **lk)
            pb, stb, _ = cv2.calcOpticalFlowPyrLK(gray[k + 1], gray[k], p1, None, **lk)
            alive &= (st[:, 0] == 1) & (stb[:, 0] == 1) & (np.linalg.norm(pb - cur, axis=2)[:, 0] < 1.0)
            cur = p1
        k += 1
        flow = float(np.median(np.linalg.norm((cur - kp)[alive, 0], axis=1))) if alive.sum() else 0.0
        if flow < MIN_FLOW and k - key < MAX_GAP and k < n - 1 and alive.sum() >= max(MIN_POINTS * 2, len(kp) // 3):
            continue
        tried += 1
        move = None                # keyframe camera -> camera k (OpenCV axes)
        a, b = kp[alive, 0], cur[alive, 0]
        if len(a) >= MIN_POINTS and flow < STILL_FLOW:
            move = np.eye(4)       # nothing moved: the camera held still
        elif len(a) >= MIN_POINTS:
            zz = kz[a[:, 1].astype(int).clip(0, h - 1), a[:, 0].astype(int).clip(0, w - 1)]
            X = np.stack([(a[:, 0] - cx) / f * zz, (a[:, 1] - cy) / f * zz, zz], 1).astype(np.float64)
            ok, rv, tv, inl = cv2.solvePnPRansac(X, b.astype(np.float64), K, None, reprojectionError=2.0,
                                                 iterationsCount=300, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
            if ok and inl is not None and len(inl) >= MIN_POINTS:
                idx = inl[:, 0]
                rv, tv = cv2.solvePnPRefineLM(X[idx], b[idx].astype(np.float64), K, None, rv, tv)
                R, _ = cv2.Rodrigues(rv)
                move = np.eye(4)
                move[:3, :3], move[:3, 3] = R, tv[:, 0]
        gap = k - key
        if move is not None:
            solved += 1
            measured[k] = True
            last_move, last_gap = move, gap
            keys[k] = keys[key] @ np.linalg.inv(move)
        else:
            # keep the last speed: the same move, scaled to this gap
            rel = np.linalg.inv(last_move)
            rot = Rotation.from_matrix(rel[:3, :3]).as_rotvec() * (gap / last_gap)
            fill = np.eye(4)
            fill[:3, :3] = Rotation.from_rotvec(rot).as_matrix()
            fill[:3, 3] = rel[:3, 3] * (gap / last_gap)
            keys[k] = keys[key] @ fill
        key = k
        near = near_at(k)
        kp = _track_points(cv2, gray[k], near)
        kz = distance(near)
        cur, alive = kp.copy(), np.ones(len(kp), bool)
        if progress:
            progress(k, n)

    # every frame: positions interpolated between keyframes, rotations slerped
    ks = sorted(keys)
    kpos = np.array([keys[i][:3, 3] for i in ks])
    krot = Rotation.from_matrix(np.array([keys[i][:3, :3] for i in ks]))
    t = np.arange(n)
    pos = np.stack([np.interp(t, ks, kpos[:, j]) for j in range(3)], 1)
    rot = Slerp(ks, krot)(np.clip(t, ks[0], ks[-1])) if len(ks) > 1 else Rotation.from_matrix(np.repeat(np.eye(3)[None], n, 0))
    q = rot.as_quat()
    q *= np.sign((q * q[:1]).sum(1, keepdims=True) + 1e-12)   # one hemisphere, so averaging works
    pos = _smooth(pos, SMOOTH_SIGMA)
    q = _smooth(q, SMOOTH_SIGMA)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    poses = np.repeat(np.eye(4)[None], n, 0)
    poses[:, :3, :3] = Rotation.from_quat(q).as_matrix()
    poses[:, :3, 3] = pos
    # the first frame is the start of the world
    poses = np.linalg.inv(poses[0]) @ poses
    return poses, measured, tried, solved


def to_gl(poses_cv):
    """Camera-to-world matrices from OpenCV axes to OpenGL axes (world turned the same way), column-major lists."""
    out = []
    for m in poses_cv:
        g = S @ m @ S
        out.append([round(float(v), 6) for v in g.T.reshape(-1)])
    return out


def read_gray(src, fps, w, h, frames):
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found on this computer")
    cmd = [ffmpeg, "-v", "error", "-nostdin", "-i", src, "-an", "-sn", "-vf",
           f"fps={fps},scale={w}:{h}:flags=area,format=gray", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg couldn't read the video: " + r.stderr.decode("utf-8", "replace")[-300:])
    g = np.frombuffer(r.stdout, np.uint8)
    got = len(g) // (w * h)
    g = g[:got * w * h].reshape(got, h, w)
    if got < frames:   # a frame short at the very end: repeat the last one
        g = np.concatenate([g, np.repeat(g[-1:], frames - got, 0)]) if got else g
    return g[:frames]


def path_file(depth_base, fov):
    return f"{depth_base}.path-fov{int(round(fov))}.json"


def run(src, depth_base, fov, meta=None):
    """Measure the path of a video whose depth is in depth_base(.bin, .json; or `meta` for the .json)."""
    t0 = time.time()
    if meta is None:
        with open(depth_base + ".json", encoding="utf-8") as fh:
            meta = json.load(fh)
    n, dw, dh = meta["frames"], meta["width"], meta["height"]
    near = np.fromfile(meta.get("bin", depth_base + ".bin"), np.uint8).reshape(n, dh, dw).astype(np.float32) / 255.0
    a = dw / dh
    w, h = (TRACK_LONG, int(round(TRACK_LONG / a / 2)) * 2) if a >= 1 else (int(round(TRACK_LONG * a / 2)) * 2, TRACK_LONG)
    gray = read_gray(src, meta["fps"], w, h, n)
    if len(gray) < 2:
        raise RuntimeError("the video is too short for a camera path")
    poses, measured, tried, solved = measure(gray, near[:len(gray)], fov)
    out = {"version": PATH_VERSION, "fps": meta["fps"], "frames": len(poses), "fov": fov, "depthC": DEPTH_C,
           "steps": tried, "solved": solved, "seconds": round(time.time() - t0, 2),
           "measured": [int(v) for v in measured], "poses": to_gl(poses)}
    dst = path_file(depth_base, fov)
    with open(dst + ".part", "w", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"))
    os.replace(dst + ".part", dst)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--depth", required=True, help="the depth file, without its extension")
    ap.add_argument("--fov", type=float, default=65)
    ap.add_argument("--parent", type=int, help="stop when this process (the server) is gone")
    a = ap.parse_args()
    if a.parent:
        from depth import watch_parent
        watch_parent(a.parent)
    no_throttle()
    try:
        out = run(a.src, a.depth, a.fov)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"t": "error", "error": str(e) or e.__class__.__name__}), flush=True)
        sys.exit(1)
    print(json.dumps({"t": "done", "steps": out["steps"], "solved": out["solved"], "seconds": out["seconds"],
                      "frames": out["frames"]}), flush=True)


if __name__ == "__main__":
    main()
