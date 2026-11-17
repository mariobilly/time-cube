"""Draws the Time Cube icon (icon.png, static/icon.png and a multi-size icon.ico).

A tilted cube made of stacked frames on a pale grey tile: thin wire planes at the back fading into a solid front
face in warm photo colours, like the cube in the reel.
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
S = 1024


def project(p, yaw=-34, pitch=22):
    x, y, z = p
    a, b = math.radians(yaw), math.radians(pitch)
    x, z = x * math.cos(a) + z * math.sin(a), -x * math.sin(a) + z * math.cos(a)
    y, z = y * math.cos(b) - z * math.sin(b), y * math.sin(b) + z * math.cos(b)
    d = 4.2
    k = d / (d - z)
    return (S / 2 + x * k * S * 0.36, S / 2 - y * k * S * 0.36 + S * 0.02)


def draw():
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    tile = Image.new("L", (S, S), 0)
    ImageDraw.Draw(tile).rounded_rectangle((24, 24, S - 24, S - 24), radius=200, fill=255)
    bg = Image.new("RGBA", (S, S), (236, 236, 233, 255))
    img.paste(bg, (0, 0), tile)
    d = ImageDraw.Draw(img, "RGBA")
    sx, sy, sz = 0.78, 0.6, 0.62
    planes = 14
    # wire planes, back to front, darker toward the front
    for k in range(planes):
        z = -sz + 2 * sz * k / (planes - 1)
        t = k / (planes - 1)
        col = (int(40 + 60 * (1 - t)), int(34 + 50 * (1 - t)), int(30 + 40 * (1 - t)), int(70 + 120 * t))
        corners = [project((-sx, -sy, z)), project((sx, -sy, z)), project((sx, sy, z)), project((-sx, sy, z))]
        d.line(corners + [corners[0]], fill=col, width=5)
        for g in range(1, 4):  # a few grid lines and diagonals
            u = -sx + 2 * sx * g / 4
            d.line([project((u, -sy, z)), project((u, sy, z))], fill=col[:3] + (col[3] // 2,), width=3)
            d.line([project((u - sx / 2, -sy, z)), project((u, sy, z))], fill=col[:3] + (col[3] // 3,), width=2)
    # the front face: a warm photo gradient with a dark round head
    front = [project((-sx, sy, sz)), project((sx, sy, sz)), project((sx, -sy, sz)), project((-sx, -sy, sz))]
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).polygon(front, fill=255)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) / S
    warm = np.stack([248 - 40 * yy, 228 - 70 * yy - 20 * xx, 196 - 80 * yy, np.full_like(xx, 255)], -1)
    hx, hy = project((-0.12, 0.08, sz))
    r = np.hypot(xx * S - hx, (yy * S - hy) * 1.05) / (S * 0.16)
    head = np.clip(1.25 - r, 0, 1)[..., None]
    warm[..., :3] = warm[..., :3] * (1 - head * 0.9) + np.array([30, 22, 20]) * head * 0.9
    face = Image.fromarray(warm.clip(0, 255).astype(np.uint8), "RGBA")
    glow = mask.filter(ImageFilter.GaussianBlur(26))
    white = Image.new("RGBA", (S, S), (255, 255, 255, 255))
    img.paste(white, (0, 0), glow.point(lambda v: int(v * 0.8)))
    img.paste(face, (0, 0), mask)
    d = ImageDraw.Draw(img, "RGBA")
    d.line(front + [front[0]], fill=(40, 34, 30, 200), width=5)
    return img


def main():
    img = draw()
    img.save(os.path.join(HERE, "icon.png"))
    img.resize((256, 256), Image.LANCZOS).save(os.path.join(HERE, "static", "icon.png"))
    img.save(os.path.join(HERE, "icon.ico"), sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("icon written")


if __name__ == "__main__":
    main()
