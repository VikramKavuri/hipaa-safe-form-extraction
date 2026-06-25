"""Generate an animated explainer GIF of the pipeline (10th-grade friendly).

Pure Pillow (no emoji fonts, no browser) so it renders identically everywhere
and animates inline on GitHub. Run:  python docs/make_explainer.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 900, 460
OUT = Path(__file__).resolve().parent / "how-it-works.gif"

# palette
BG = (247, 249, 252)
INK = (30, 41, 59)
MUTE = (100, 116, 139)
BLUE = (37, 99, 235)
INDIGO = (79, 70, 229)
GREEN = (22, 163, 74)
AMBER = (217, 119, 6)
CARD = (255, 255, 255)
CARD_ON = (238, 242, 255)
BORDER = (203, 213, 225)
PAPER = (255, 255, 255)


def _font(size: int, *names: str):
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            continue
    return ImageFont.load_default()


F_TITLE = _font(30, "arialbd.ttf", "DejaVuSans-Bold.ttf")
F_CARD = _font(18, "arialbd.ttf", "DejaVuSans-Bold.ttf")
F_SUB = _font(14, "arial.ttf", "DejaVuSans.ttf")
F_TINY = _font(12, "arial.ttf", "DejaVuSans.ttf")

# 5 pipeline stages: (title, subtitle, accent)
STAGES = [
    ("Messy form", "handwritten scan", AMBER),
    ("Clean it up", "straighten + sharpen", BLUE),
    ("AI reads it", "Qwen2.5-VL", INDIGO),
    ("Find ticked boxes", "tight crop + check", BLUE),
    ("Neat data", "structured fields", GREEN),
]

CARD_W, CARD_H = 150, 170
TOP = 150
XS = [30 + i * 174 for i in range(5)]  # left x of each card
CX = [x + CARD_W // 2 for x in XS]
CY = TOP + CARD_H // 2


def _center(d, cx, y, text, font, fill):
    w = d.textlength(text, font=font)
    d.text((cx - w / 2, y), text, font=font, fill=fill)


# ---- little drawn icons --------------------------------------------------
def icon_paper(d, cx, cy, messy):
    x0, y0 = cx - 30, cy - 38
    d.rounded_rectangle([x0, y0, x0 + 60, y0 + 76], radius=6, fill=PAPER, outline=BORDER, width=2)
    for i, yy in enumerate(range(y0 + 14, y0 + 70, 11)):
        wob = (3 if messy else 0) * (1 if i % 2 else -1)
        d.line([x0 + 10 + wob, yy, x0 + 50 + wob, yy], fill=(120, 130, 150), width=2)
    if not messy:  # sparkle
        sx, sy = x0 + 58, y0 - 4
        d.line([sx - 5, sy, sx + 5, sy], fill=AMBER, width=2)
        d.line([sx, sy - 5, sx, sy + 5], fill=AMBER, width=2)


def icon_ai(d, cx, cy, scan_t):
    x0, y0 = cx - 32, cy - 30
    d.line([cx, y0 - 10, cx, y0], fill=INDIGO, width=2)
    d.ellipse([cx - 3, y0 - 14, cx + 3, y0 - 8], fill=INDIGO)
    d.rounded_rectangle([x0, y0, x0 + 64, y0 + 56], radius=10, fill=(237, 233, 254), outline=INDIGO, width=2)
    for ex in (cx - 14, cx + 14):
        d.ellipse([ex - 7, cy - 8, ex + 7, cy + 6], fill="white", outline=INDIGO, width=2)
        d.ellipse([ex - 3, cy - 4, ex + 3, cy + 2], fill=INDIGO)
    d.line([cx - 12, y0 + 48, cx + 12, y0 + 48], fill=INDIGO, width=2)
    if scan_t is not None:  # sweeping scan line
        yy = int(y0 + 6 + scan_t * 44)
        d.line([x0 + 4, yy, x0 + 60, yy], fill=GREEN, width=3)


def icon_checks(d, cx, cy, lit):
    for i, yy in enumerate(range(cy - 30, cy + 22, 22)):
        x0 = cx - 26
        d.rounded_rectangle([x0, yy, x0 + 18, yy + 18], radius=3, fill="white", outline=BORDER, width=2)
        if i == 1 and lit:  # the ticked one
            d.line([x0 + 3, yy + 10, x0 + 7, yy + 15], fill=GREEN, width=3)
            d.line([x0 + 7, yy + 15, x0 + 15, yy + 3], fill=GREEN, width=3)
        d.line([x0 + 26, yy + 4, x0 + 40, yy + 4], fill=(150, 160, 175), width=2)
        d.line([x0 + 26, yy + 12, x0 + 34, yy + 12], fill=(180, 190, 205), width=2)


def icon_table(d, cx, cy, rows):
    x0, y0 = cx - 34, cy - 34
    d.rounded_rectangle([x0, y0, x0 + 68, y0 + 68], radius=6, fill="white", outline=BORDER, width=2)
    d.rectangle([x0, y0, x0 + 68, y0 + 16], fill=(220, 252, 231))
    d.line([x0 + 34, y0, x0 + 34, y0 + 68], fill=BORDER, width=1)
    for r in range(4):
        yy = y0 + 16 + r * 13
        d.line([x0, yy, x0 + 68, yy], fill=BORDER, width=1)
        if r < rows:
            d.line([x0 + 6, yy + 6, x0 + 28, yy + 6], fill=GREEN, width=3)
            d.line([x0 + 40, yy + 6, x0 + 62, yy + 6], fill=(90, 110, 140), width=3)


def icon_lock(d, x, y, c=GREEN):
    d.arc([x, y - 8, x + 16, y + 8], 180, 360, fill=c, width=3)
    d.rounded_rectangle([x - 3, y + 2, x + 19, y + 20], radius=3, fill=c)
    d.ellipse([x + 6, y + 8, x + 10, y + 12], fill="white")


DRAW = [
    lambda d, c, t: icon_paper(d, c, CY, messy=True),
    lambda d, c, t: icon_paper(d, c, CY, messy=False),
    lambda d, c, t: icon_ai(d, c, CY, scan_t=t),
    lambda d, c, t: icon_checks(d, c, CY, lit=(t is not None)),
    lambda d, c, t: icon_table(d, c, CY, rows=int((t or 0) * 4) + 1),
]


def frame(f: int, total: int) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    _center(d, W // 2, 26, "How it works:  paper form  →  neat data", F_TITLE, INK)
    _center(d, W // 2, 64, "messy handwritten forms turned into clean, organized data by an AI", F_SUB, MUTE)

    active = min(4, int(f / total * 5))
    sub = (f / total * 5) - active  # 0..1 progress within the active stage

    # connecting arrows
    for i in range(4):
        x1, x2 = XS[i] + CARD_W, XS[i + 1]
        ay = CY
        d.line([x1 + 4, ay, x2 - 8, ay], fill=BORDER, width=3)
        d.polygon([(x2 - 8, ay - 5), (x2 - 8, ay + 5), (x2, ay)], fill=BORDER)

    # cards
    for i, (title, subt, accent) in enumerate(STAGES):
        on = i <= active
        x0 = XS[i]
        lift = 6 if i == active else 0
        if i == active:  # pulsing glow ring
            pw = 3 + int(2 + 2 * abs(0.5 - sub) * 2)
            d.rounded_rectangle(
                [x0 - 4, TOP - 4 - lift, x0 + CARD_W + 4, TOP + CARD_H + 4 - lift],
                radius=18, outline=accent, width=pw,
            )
        d.rounded_rectangle(
            [x0, TOP - lift, x0 + CARD_W, TOP + CARD_H - lift],
            radius=14, fill=CARD_ON if i == active else (CARD if on else BG),
            outline=accent if on else BORDER, width=2,
        )
        scan_t = sub if i == active else None
        DRAW[i](d, CX[i], scan_t)
        _center(d, CX[i], TOP + CARD_H - 52 - lift, title, F_CARD, INK if on else MUTE)
        _center(d, CX[i], TOP + CARD_H - 30 - lift, subt, F_TINY, accent if on else MUTE)
        d.ellipse([CX[i] - 9, TOP - 26, CX[i] + 9, TOP - 8], fill=accent if on else BORDER)
        _center(d, CX[i], TOP - 25, str(i + 1), F_TINY, "white")

    # travelling packet dot along the baseline
    if active == 0:
        px = CX[0]
    else:
        px = CX[active - 1] + (CX[active] - CX[active - 1]) * sub
    d.ellipse([px - 7, CY + CARD_H // 2 + 18, px + 7, CY + CARD_H // 2 + 32], fill=GREEN)

    # privacy footer (lock + text, centered as a group so it fits any length)
    fy = H - 46
    text = "HIPAA-safe  •  100% local  •  your data never leaves your computer"
    tw = d.textlength(text, font=F_SUB)
    lock_w, gap, pad = 22, 14, 24
    content_w = lock_w + gap + tw
    sx = (W - content_w) / 2
    d.rounded_rectangle([sx - pad, fy, sx + content_w + pad, fy + 34], radius=17, fill=(220, 252, 231))
    icon_lock(d, sx + 3, fy + 7)
    d.text((sx + lock_w + gap, fy + 9), text, font=F_SUB, fill=GREEN)
    return img


def main() -> None:
    total = 60
    frames = [frame(f, total) for f in range(total)]
    frames[0].save(
        OUT, save_all=True, append_images=frames[1:], duration=110, loop=0, optimize=True, disposal=2
    )
    print(f"wrote {OUT}  ({total} frames)")


if __name__ == "__main__":
    main()
