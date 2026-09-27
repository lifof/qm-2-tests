"""Lettering (speech bubbles, captions, SFX) and stacking panels into a vertical webtoon strip.

Text is drawn here rather than by the image model so it is always legible and editable.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import List, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont

from .models import PanelPlan

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/comicbd.ttf",
]

GUTTER = 70  # vertical white space between panels
SCENE_GUTTER = 220  # bigger gap when the location changes
SLICE_HEIGHT = 1280  # typical webtoon upload slice (800x1280)


def load_font(size: int, font_path: Optional[str] = None) -> ImageFont.FreeTypeFont:
    for path in ([font_path] if font_path else []) + FONT_CANDIDATES:
        if path and Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> List[str]:
    lines: List[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split():
            trial = f"{current} {word}".strip()
            if draw.textlength(trial, font=font) <= max_width or not current:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def _text_block(draw, text, font, max_width, spacing=6):
    lines = _wrap(draw, text, font, max_width)
    body = "\n".join(lines)
    left, top, right, bottom = draw.multiline_textbbox((0, 0), body, font=font, spacing=spacing, align="center")
    return body, right - left, bottom - top


def _draw_bubble(img: Image.Image, x: int, y: int, text: str, kind: str, font, max_width: int, tail_to_right: bool,
                 occupied: Optional[list] = None) -> int:
    """Draw one bubble with its top-left at (x, y); returns its bottom y."""
    draw = ImageDraw.Draw(img)
    if kind == "shout":
        text = text.upper()
    body, tw, th = _text_block(draw, text, font, max_width)
    pad_x, pad_y = 26, 18
    w, h = tw + 2 * pad_x, th + 2 * pad_y
    x = max(8, min(x, img.width - w - 8))
    box = (x, y, x + w, y + h)
    if occupied is not None:
        occupied.append((x, y, x + w, y + h + 34))
    outline = (20, 20, 20)
    fill = (255, 255, 255)

    if kind == "thought":
        draw.ellipse(box, fill=fill, outline=outline, width=3)
        cx = x + (w * 3 // 4 if tail_to_right else w // 4)
        for i, r in enumerate((10, 7, 4)):
            cy = y + h + 8 + i * 14
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=outline, width=2)
    else:
        radius = min(h // 2, 40)
        tail_x = x + (w * 2 // 3 if tail_to_right else w // 3)
        tail = [(tail_x - 14, y + h - 4), (tail_x + 14, y + h - 4), (tail_x + (22 if tail_to_right else -22), y + h + 28)]
        width = 5 if kind == "shout" else 3
        draw.polygon(tail, fill=fill, outline=outline, width=width)
        draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
        # re-cover the seam between tail and bubble
        draw.polygon([(tail_x - 11, y + h - 6), (tail_x + 11, y + h - 6), (tail_x, y + h + 2)], fill=fill)
        if kind == "whisper":
            for dx in range(x + 10, x + w - 10, 16):
                draw.line((dx, y + 2, dx + 8, y + 2), fill=fill, width=4)
                draw.line((dx, y + h - 2, dx + 8, y + h - 2), fill=fill, width=4)
    text_color = (90, 90, 90) if kind == "whisper" else (15, 15, 15)
    draw.multiline_text((x + w / 2, y + h / 2), body, font=font, fill=text_color, anchor="mm", align="center", spacing=6)
    return y + h + 34


def _draw_caption(img: Image.Image, text: str, font, max_width: int, occupied: Optional[list] = None) -> int:
    draw = ImageDraw.Draw(img)
    body, tw, th = _text_block(draw, text, font, max_width)
    pad = 16
    box = (12, 12, 12 + tw + 2 * pad, 12 + th + 2 * pad)
    if occupied is not None:
        occupied.append(box)
    draw.rectangle(box, fill=(250, 244, 220), outline=(20, 20, 20), width=3)
    draw.multiline_text((box[0] + pad, box[1] + pad), body, font=font, fill=(20, 20, 20), spacing=6)
    return box[3] + 14


SYSTEM_FILL = (14, 28, 64, 225)
SYSTEM_BORDER = (110, 200, 255)
SYSTEM_TEXT = (226, 240, 255)


def _system_size(draw, text: str, font, max_width: int, pad: int = 22) -> tuple[str, int, int]:
    lines = _wrap(draw, text.strip(), font, max_width - 2 * pad)
    body = "\n".join(lines)
    left, top, right, bottom = draw.multiline_textbbox((0, 0), body, font=font, spacing=8)
    return body, right - left + 2 * pad, bottom - top + 2 * pad


def _draw_system(img: Image.Image, y: int, text: str, font, max_width: int, occupied: Optional[list] = None) -> int:
    """A game-UI window (quests, stats, notifications): translucent blue box, centred. Returns its bottom y."""
    draw = ImageDraw.Draw(img)
    body, w, h = _system_size(draw, text, font, max_width)
    x = (img.width - w) // 2
    if occupied is not None:
        occupied.append((x, y, x + w, y + h))
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rounded_rectangle((x, y, x + w, y + h), radius=14, fill=SYSTEM_FILL, outline=SYSTEM_BORDER, width=3)
    od.line((x + 14, y + 7, x + w - 14, y + 7), fill=SYSTEM_BORDER + (160,), width=2)  # UI "title bar" accent
    img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))
    ImageDraw.Draw(img).multiline_text((x + 22, y + 22), body, font=font, fill=SYSTEM_TEXT, spacing=8)
    return y + h + 20


def _overlaps(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _draw_sfx(img: Image.Image, text: str, font_path: Optional[str], occupied: Sequence[tuple] = ()) -> None:
    """Tilted SFX lettering in a corner that doesn't cover bubbles, captions or windows."""
    text = text.upper()
    font = load_font(max(40, img.width // 12), font_path)
    probe = ImageDraw.Draw(img)
    left, top, right, bottom = probe.textbbox((0, 0), text, font=font, stroke_width=5)
    layer = Image.new("RGBA", (right - left + 20, bottom - top + 20), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((10 - left, 10 - top), text, font=font, fill=(255, 214, 0, 255), stroke_width=5,
                               stroke_fill=(20, 20, 20, 255))
    layer = layer.rotate(7, expand=True, resample=Image.BICUBIC)
    if layer.width > img.width - 20:  # very long SFX: scale it down to fit
        ratio = (img.width - 20) / layer.width
        layer = layer.resize((int(layer.width * ratio), int(layer.height * ratio)), Image.LANCZOS)
    w, h = layer.size
    margin = 18
    candidates = [(img.width - w - margin, img.height - h - margin), (margin, img.height - h - margin),
                  (img.width - w - margin, int(img.height * 0.45)), (margin, int(img.height * 0.45)),
                  (img.width - w - margin, margin)]
    x, y = next(((cx, cy) for cx, cy in candidates if not any(_overlaps((cx, cy, cx + w, cy + h), o) for o in occupied)),
                candidates[0])
    base = img.convert("RGBA")
    base.alpha_composite(layer, (max(0, x), max(0, y)))
    img.paste(base.convert("RGB"))


def _letter_on(img: Image.Image, panel: PanelPlan, width: int, font, occupied: list) -> List:
    """Draw captions/bubbles/windows that fit on the art; return the lines that don't."""
    max_text_w = int(width * 0.52)
    system_w = int(width * 0.86)
    measure = ImageDraw.Draw(img)
    y = 16
    if panel.narration.strip():
        y = _draw_caption(img, panel.narration.strip(), font, int(width * 0.8), occupied)
    # Bubbles alternate left/right, stacked downward, reading order top-left first.
    overflow: List = []
    bubble_i = 0
    for line in panel.dialogue:
        if overflow:
            overflow.append(line)  # keep reading order once we've spilled over
            continue
        if line.kind == "system":
            if y + _system_size(measure, line.text, font, system_w)[2] > img.height * 0.92:
                overflow.append(line)
            else:
                y = _draw_system(img, y, line.text, font, system_w, occupied)
            continue
        left = bubble_i % 2 == 0
        bubble_i += 1
        _, tw, th = _text_block(measure, line.text, font, max_text_w)
        if y + th + 60 > img.height * 0.8:
            overflow.append(line)
            continue
        y = _draw_bubble(img, 24 if left else int(width * 0.42), y, line.text, line.kind, font, max_text_w,
                         tail_to_right=left, occupied=occupied) - 10
    return overflow


def letter_panel(art: Image.Image, panel: PanelPlan, width: int, font_path: Optional[str] = None) -> Image.Image:
    """Resize a generated panel to the strip width and draw its text on it."""
    art = art.convert("RGB")
    base = art.resize((width, round(art.height * width / art.width)), Image.LANCZOS)
    # Try the normal text size first; if something doesn't fit on the art, try slightly smaller text
    # before spilling over into white space below the panel.
    for scale in (1.0, 0.85):
        font = load_font(max(18, int(width // 30 * scale)), font_path)
        img, occupied = base.copy(), []
        overflow = _letter_on(img, panel, width, font, occupied)
        if not overflow:
            break

    if overflow:
        system_w, max_text_w = int(width * 0.86), int(width * 0.52)
        measure = ImageDraw.Draw(img)
        height = 40 + sum(_system_size(measure, l.text, font, system_w)[2] + 30 if l.kind == "system" else 200
                          for l in overflow)
        extra = Image.new("RGB", (width, height), (255, 255, 255))
        ey = 20
        for i, line in enumerate(overflow):
            if line.kind == "system":
                ey = _draw_system(extra, ey, line.text, font, system_w)
                continue
            text = f"{line.speaker}: {line.text}" if line.speaker else line.text
            ey = _draw_bubble(extra, 24 if i % 2 == 0 else int(width * 0.42), ey, text,
                              line.kind, font, max_text_w, tail_to_right=i % 2 == 0)
        extra = extra.crop((0, 0, width, ey))
        combined = Image.new("RGB", (width, img.height + extra.height), (255, 255, 255))
        combined.paste(img, (0, 0))
        combined.paste(extra, (0, img.height))
        img = combined

    if panel.sfx.strip():
        _draw_sfx(img, panel.sfx.strip(), font_path, occupied)
    return img


def build_strip(panels: Sequence[Image.Image], plans: Sequence[PanelPlan], width: int) -> Image.Image:
    gaps = [0] + [
        SCENE_GUTTER if plans[i].location.strip().lower() != plans[i - 1].location.strip().lower() else GUTTER
        for i in range(1, len(plans))
    ]
    total = sum(p.height for p in panels) + sum(gaps) + 2 * GUTTER
    strip = Image.new("RGB", (width, total), (255, 255, 255))
    y = GUTTER
    for gap, panel in zip(gaps, panels):
        y += gap
        strip.paste(panel, (0, y))
        y += panel.height
    return strip


def slice_strip(strip: Image.Image, out_dir: Path, prefix: str) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, top in enumerate(range(0, strip.height, SLICE_HEIGHT)):
        piece = strip.crop((0, top, strip.width, min(top + SLICE_HEIGHT, strip.height)))
        path = out_dir / f"{prefix}_{i + 1:03d}.jpg"
        piece.save(path, quality=92)
        paths.append(path)
    return paths


def write_reader(out_file: Path, title: str, slice_paths: Sequence[Path], width: int) -> None:
    rel = [p.relative_to(out_file.parent).as_posix() for p in slice_paths]
    imgs = "\n".join(f'<img src="{html.escape(r)}" alt="">' for r in rel)
    out_file.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{ margin: 0; background: #1b1b1f; color: #eee; font-family: system-ui, sans-serif; }}
h1 {{ text-align: center; font-size: 1.2rem; padding: 1rem; }}
main {{ max-width: {width}px; margin: 0 auto; background: #fff; }}
img {{ display: block; width: 100%; height: auto; }}
</style></head>
<body><h1>{html.escape(title)}</h1><main>
{imgs}
</main></body></html>
""",
        encoding="utf-8",
    )
