"""Camera path tests: the measuring is checked against a rendered scene whose camera path is known.

    python tests/test_path.py

A small ray tracer draws a room with boxes in it (random block textures, so there are corners to follow) along a
known camera path. The path is then measured twice: from the scene's true depth (the solver alone), and from the
rendered video through the whole chain, Depth Anything included (needs the NVIDIA card; skipped without it). The
error is the distance between the measured and the true camera positions, after lining the two paths up (Umeyama:
scale, turn, shift), as a percentage of the true path's length.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
TMP = tempfile.mkdtemp(prefix="timecube-path-test-")
os.environ["TIME_CUBE_DATA"] = os.path.join(TMP, "data")
sys.path.insert(0, APP)
sys.path.insert(0, HERE)
import path as P  # noqa: E402
import server  # noqa: E402

W, H, FOV, FPS = 640, 360, 65.0, 24
SOLVER_LIMIT = 2.0     # % of the path length, with the true depth
PIPELINE_LIMIT = 10.0  # % with Depth Anything's depth


# ---- a tiny ray-traced scene (OpenCV axes: x right, y down, z forward) ----------------------------------------
ROOM = (np.array([-5.0, -2.6, -2.0]), np.array([5.0, 2.4, 11.5]))
BOXES = [(np.array([-3.6, 0.6, 5.0]), np.array([-2.0, 2.4, 6.6])),
         (np.array([1.4, -0.4, 6.5]), np.array([3.2, 2.4, 8.0])),
         (np.array([-1.0, 1.4, 8.5]), np.array([0.6, 2.4, 9.6])),
         (np.array([3.6, -2.6, 3.2]), np.array([5.0, 2.4, 4.6])),
         (np.array([-0.6, -2.6, 9.8]), np.array([1.6, -1.2, 11.5]))]


def true_path(n):
    """Camera-to-world (OpenCV axes): forward 3 units with a sideways swing, turning 22° and nodding a little."""
    from scipy.spatial.transform import Rotation
    out = []
    for i in range(n):
        s = i / (n - 1)
        pos = np.array([0.7 * np.sin(np.pi * s) - 0.2 * s, -0.15 * np.sin(2 * np.pi * s), 3.0 * s])
        R = Rotation.from_euler("yxz", [22 * s - 4, 3 * np.sin(np.pi * s), 1.5 * np.sin(2 * np.pi * s)],
                                degrees=True).as_matrix()
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = R, pos
        out.append(m)
    return np.array(out)


@np.errstate(over="ignore")       # the hash wraps around on purpose
def _hash(*ints):
    v = np.zeros_like(ints[0], dtype=np.uint64)
    for k, a in enumerate(ints):
        v ^= (a.astype(np.int64).astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15) * np.uint64(k + 1))
        v = (v ^ (v >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        v = (v ^ (v >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        v ^= v >> np.uint64(31)
    return (v & np.uint64(0xFFFFFF)).astype(np.float64) / float(0xFFFFFF)


def render(pose):
    """One frame: RGB (H, W, 3) uint8 and the depth along the view (H, W)."""
    K = P.intrinsics(W, H, FOV)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
    R, o = pose[:3, :3], pose[:3, 3]
    dw = d @ R.T
    inv = 1.0 / np.where(np.abs(dw) < 1e-9, 1e-9, dw)
    best = np.full(len(dw), np.inf)
    face = np.zeros(len(dw), int)
    # the room, seen from inside: where the ray leaves it
    t1, t2 = (ROOM[0] - o) * inv, (ROOM[1] - o) * inv
    tfar = np.maximum(t1, t2)
    best = tfar.min(1)
    face = tfar.argmin(1) + 3 * (dw[np.arange(len(dw)), tfar.argmin(1)] > 0)
    for bi, (lo, hi) in enumerate(BOXES):
        t1, t2 = (lo - o) * inv, (hi - o) * inv
        tn, tf = np.minimum(t1, t2), np.maximum(t1, t2)
        tin, tout = tn.max(1), tf.min(1)
        hit = (tin < tout) & (tin > 1e-3) & (tin < best)
        ax = tn.argmax(1)
        best = np.where(hit, tin, best)
        face = np.where(hit, 10 + bi * 6 + ax + 3 * (dw[np.arange(len(dw)), ax] < 0), face)
    p = o + dw * best[:, None]
    # texture: blocks of colour on each face, with finer speckle, plus simple shading
    fa = np.where(face < 10, face % 3, (face - 10) % 6 % 3)
    a_ = np.where(fa == 0, p[:, 1], p[:, 0])
    b_ = np.where(fa == 2, p[:, 1], p[:, 2])
    ia, ib = np.floor(a_ / 0.35).astype(np.int64), np.floor(b_ / 0.35).astype(np.int64)
    hue = _hash(ia, ib, face)
    base = np.stack([0.35 + 0.6 * _hash(ia, ib, face + 101), 0.35 + 0.6 * hue, 0.35 + 0.6 * _hash(ia, ib, face + 7)], 1)
    fine = _hash(np.floor(a_ / 0.07).astype(np.int64), np.floor(b_ / 0.07).astype(np.int64), face + 33)
    shade = np.array([0.95, 0.8, 0.65, 0.95, 0.75, 0.6])[np.where(face < 10, face % 6, (face - 10) % 6)]
    col = base * (0.7 + 0.3 * fine[:, None]) * shade[:, None] * (1.0 / (1.0 + 0.04 * best[:, None]))
    rgb = (np.clip(col, 0, 1) * 255).astype(np.uint8).reshape(H, W, 3)
    depth = (best * d[:, 2]).reshape(H, W)   # along the view axis
    return rgb, depth


def umeyama_error(est, ref):
    """RMS distance between two point paths after the best scale, turn and shift, in % of ref's length."""
    mu_e, mu_r = est.mean(0), ref.mean(0)
    e, r = est - mu_e, ref - mu_r
    U, D, Vt = np.linalg.svd(r.T @ e / len(est))
    Sg = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Sg[2, 2] = -1
    R = U @ Sg @ Vt
    s = np.trace(np.diag(D) @ Sg) / max((e ** 2).sum(1).mean(), 1e-12)
    aligned = (s * (R @ e.T)).T + mu_r
    rmse = np.sqrt(((aligned - ref) ** 2).sum(1).mean())
    length = np.linalg.norm(np.diff(ref, axis=0), axis=1).sum()
    return 100 * rmse / length, s


def scene(n=96):
    poses = true_path(n)
    frames, depths = zip(*(render(m) for m in poses))
    return poses, np.array(frames), np.array(depths)


def gray_of(frames):
    return (np.asarray(frames, np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)).astype(np.uint8)


def relative(poses):
    return np.linalg.inv(poses[0]) @ poses


class SolverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        t = time.time()
        cls.gt, cls.frames, cls.depths = scene()
        print(f"\n  rendered {len(cls.frames)} frames in {time.time() - t:.1f} s", flush=True)

    def test_true_depth(self):
        """With the true depth, the measured path follows the real one closely."""
        near = np.clip((1.0 / np.maximum(self.depths, 1.0) - P.DEPTH_C) / (1 - P.DEPTH_C), 0, 1)
        t = time.time()
        poses, measured, tried, solved = P.measure(gray_of(self.frames), near, FOV)
        took = time.time() - t
        est, ref = relative(poses)[:, :3, 3], relative(self.gt)[:, :3, 3]
        err, scale = umeyama_error(est, ref)
        raw = 100 * np.sqrt(((est - ref) ** 2).sum(1).mean()) / np.linalg.norm(np.diff(ref, axis=0), axis=1).sum()
        turn = np.degrees(np.arccos(np.clip((np.trace(relative(poses)[-1, :3, :3].T @ relative(self.gt)[-1, :3, :3]) - 1) / 2, -1, 1)))
        print(f"  true depth: {solved} of {tried} steps, {took:.2f} s, error {err:.2f} % of the path "
              f"(no alignment: {raw:.2f} %, scale {scale:.3f}), last turn off by {turn:.2f}°", flush=True)
        self.assertEqual(solved, tried)
        self.assertLess(err, SOLVER_LIMIT)
        self.assertLess(raw, 2 * SOLVER_LIMIT)     # the true depth's unit is the path's: no scale needed
        self.assertLess(turn, 1.5)

    def test_still_camera(self):
        """A camera that doesn't move gives a path that stays put."""
        g = np.repeat(gray_of(self.frames[:1]), 30, 0)
        near = np.repeat(np.clip((1.0 / np.maximum(self.depths[:1], 1.0) - P.DEPTH_C) / (1 - P.DEPTH_C), 0, 1), 30, 0)
        poses, *_ = P.measure(g, near, FOV)
        self.assertLess(np.abs(poses[:, :3, 3]).max(), 1e-6)
        self.assertLess(np.abs(poses[:, :3, :3] - np.eye(3)).max(), 1e-6)

    def test_gl_axes(self):
        """The file's matrices are OpenGL camera-to-world, column-major: a step forward is -z."""
        m = np.eye(4)
        m[2, 3] = 1.0        # OpenCV: one unit forward
        g = np.array(P.to_gl([m])[0]).reshape(4, 4).T
        self.assertAlmostEqual(g[2, 3], -1.0)
        self.assertAlmostEqual(g[1, 1], 1.0)

    @unittest.skipUnless(__import__("test_depth").gpu_ready(), "no NVIDIA card or Depth Anything model here")
    def test_whole_chain(self):
        """Video -> Depth Anything -> camera path, against the true path."""
        clip = os.path.join(TMP, "room walk.mp4")
        ff = server.find_tool("ffmpeg")
        proc = subprocess.Popen([ff, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                 "-r", str(FPS), "-i", "pipe:0", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
                                 clip], stdin=subprocess.PIPE)
        proc.stdin.write(self.frames.tobytes())
        proc.stdin.close()
        self.assertEqual(proc.wait(), 0)
        base = os.path.join(TMP, "room walk")
        r = subprocess.run([server.python_exe(), "-u", os.path.join(APP, "depth.py"), "--src", clip, "--out", base,
                            "--fps", str(FPS)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout[-500:] + r.stderr[-500:])
        out = P.run(clip, base, FOV)
        poses = np.array(out["poses"]).reshape(-1, 4, 4).transpose(0, 2, 1)
        poses = P.S @ poses @ P.S            # back to OpenCV axes, like the true path
        est, ref = relative(poses)[:, :3, 3], relative(self.gt)[:, :3, 3]
        err, scale = umeyama_error(est, ref)
        print(f"  Depth Anything: {out['solved']} of {out['steps']} steps, {out['seconds']} s, error {err:.2f} % of "
              f"the path (scale {scale:.3f})", flush=True)
        self.assertEqual(out["frames"], len(self.gt))
        self.assertGreaterEqual(out["solved"], 0.9 * out["steps"])
        self.assertLess(err, PIPELINE_LIMIT)
        np.save(os.path.join(TMP, "paths.npy"), np.stack([est, ref]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
