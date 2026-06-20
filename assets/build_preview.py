"""Render assets/social-preview.png (1280x640) — the GitHub social card.

Pillow + system fonts. Run:  python assets/build_preview.py
SVG can't be rasterised on this box (no libcairo), so the card is drawn directly.
"""
import math
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1280, 640
BG = (10, 14, 26)
PANEL = (16, 22, 38)
BORDER = (32, 40, 58)
CYAN = (45, 212, 255)
BLUE = (33, 150, 243)
WHITE = (247, 250, 255)
GRAY = (138, 160, 184)
GOLD = (255, 199, 64)
PINK = (255, 64, 129)
LBLUE = (207, 238, 255)


def font(names, size):
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            pass
    return ImageFont.load_default()


BOLD = font(["segoeuib.ttf", "arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf"], 132)
REG = font(["segoeui.ttf", "arial.ttf", "C:/Windows/Fonts/segoeui.ttf"], 34)
MONO = font(["consola.ttf", "cour.ttf", "C:/Windows/Fonts/consola.ttf"], 30)

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)
d.rounded_rectangle([6, 6, W - 7, H - 7], radius=30, outline=BORDER, width=2)


def magnifier(cx, cy, R, sw):
    d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=BLUE, width=sw)          # lens ring
    ax, ay = cx + R * math.cos(math.radians(45)), cy + R * math.sin(math.radians(45))
    bx, by = ax + 40, ay + 40                                                    # handle
    d.line([(ax, ay), (bx, by)], fill=BLUE, width=sw)
    d.ellipse([bx - sw / 2, by - sw / 2, bx + sw / 2, by + sw / 2], fill=BLUE)
    base = cy + 44                                                               # bar chart
    for x0, h in [(cx - 36, 40), (cx - 12, 64), (cx + 12, 50)]:
        d.rounded_rectangle([x0, base - h, x0 + 18, base], radius=4, fill=LBLUE)
    pts = [(cx - 50, cy + 30), (cx - 16, cy + 4), (cx + 14, cy - 6), (cx + 46, cy - 38)]
    d.line(pts, fill=GOLD, width=11, joint="curve")                             # rising trend
    for px, py in (pts[0], pts[-1]):
        d.ellipse([px - 5.5, py - 5.5, px + 5.5, py + 5.5], fill=GOLD)
    ex, ey = pts[-1]
    d.ellipse([ex - 13, ey - 13, ex + 13, ey + 13], fill=PINK)                  # data point


R, SW, yc = 94, 22, 244
wordW = d.textlength("firstlook", font=BOLD)
markW = 2 * R + 50
gap = 60
startX = (W - (markW + gap + wordW)) / 2

magnifier(startX + R, yc, R, SW)
wx = startX + markW + gap
d.text((wx, yc), "first", font=BOLD, fill=WHITE, anchor="lm")
d.text((wx + d.textlength("first", font=BOLD), yc), "look", font=BOLD, fill=BLUE, anchor="lm")

d.text((W / 2, yc + R + 80), "the first look at your data", font=REG, fill=GRAY, anchor="mm")

pip = "pip install firstlook"
pw = d.textlength(pip, font=MONO)
px0, py0 = (W - pw) / 2 - 22, yc + R + 124
d.rounded_rectangle([px0, py0, px0 + pw + 44, py0 + 56], radius=14, fill=PANEL, outline=BORDER, width=1)
d.text((W / 2, py0 + 28), pip, font=MONO, fill=CYAN, anchor="mm")

out = os.path.join(HERE, "social-preview.png")
img.save(out, "PNG")
print("wrote", out, "->", os.path.getsize(out), "bytes", img.size)
