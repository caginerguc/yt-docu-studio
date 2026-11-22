"""MakroKontext V3 — Thumbnail kompozitörü (Pillow + numpy, tamamen ücretsiz/offline).

Neden AI'dan metin istemiyoruz: Flux yazıyı tutarsız çizer, umlaut'ları bozar.
Arka planı AI/arşiv görseli sağlar; başlığı, rengi, vinyeti ve ışığı burada kendimiz basarız:

  1. Kırp + renk derecelendirme (shadow/highlight tonlama)
  2. Odak noktasına spot ışık + kenar karartma (vignette)
  3. Metin okunurlüğü için alt/sol gradyan
  4. Otomatik sığdırılan dev başlık (kalın kontur + gölge, son kelime vurgulu)
  5. Etiket bandı + marka damgası + ince film greni
  6. 3 varyant (renk / yerleşim) — YouTube'a en iyisi yüklenir
"""
from __future__ import annotations

import argparse
import itertools
import logging
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from src import config

log = logging.getLogger("makrokontext.thumbnail")

W, H = config.THUMB_W, config.THUMB_H

PALETTES = (
    {"fill": (255, 230, 0),   "accent": (225, 6, 0),    "tint": (255, 150, 40)},   # sarı / kırmızı
    {"fill": (255, 255, 255), "accent": (255, 59, 31),  "tint": (90, 160, 255)},   # beyaz / turuncu-kırmızı
    {"fill": (0, 229, 255),   "accent": (255, 230, 0),  "tint": (0, 200, 200)},    # camgöbeği / sarı
)
# (odak x, odak y, yerleşim, max metin genişliği oranı, max metin yüksekliği oranı)
LAYOUTS = (
    (0.64, 0.40, "left",   0.62, 0.50),
    (0.50, 0.38, "center", 0.92, 0.40),
    (0.68, 0.46, "left",   0.58, 0.58),
)


# --------------------------------------------------------------------------- #
# Yardımcılar
# --------------------------------------------------------------------------- #
def find_font_path() -> Path | None:
    for p in config.THUMB_FONT_CANDIDATES:
        if Path(p).is_file():
            return Path(p)
    return None


def load_font(size: int, path: Path | None = None) -> ImageFont.FreeTypeFont:
    path = path or find_font_path()
    if path:
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def sanitize_headline(text: str, max_words: int = 4, max_chars: int = 30) -> str:
    """'VERBOTENE AKTE: Was das FBI ...' -> 'VERBOTENE AKTE' (ilk 2-4 kelime, büyük harf)."""
    head = text.split(":")[0]
    head = re.sub(r"[^\w\s?!.\-ÄÖÜäöüß]", "", head, flags=re.UNICODE)
    words = head.upper().split()[:max_words]
    out = " ".join(words)
    while len(out) > max_chars and len(words) > 1:
        words.pop()
        out = " ".join(words)
    return out or "GEHEIMAKTE"


def _cover_resize(img: Image.Image, focus: tuple[float, float]) -> Image.Image:
    img = img.convert("RGB")
    scale = max(W / img.width, H / img.height)
    nw, nh = max(W, round(img.width * scale)), max(H, round(img.height * scale))
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left = int(min(max(focus[0] * nw - W / 2, 0), nw - W))
    top = int(min(max(focus[1] * nh - H / 2, 0), nh - H))
    return img.crop((left, top, left + W, top + H))


def procedural_background(seed: int = 0) -> Image.Image:
    """Test/demo için duman dokulu karanlık arka plan + silüet (gerçek akışta AI görseli gelir)."""
    rng = np.random.default_rng(seed)
    base = np.zeros((H, W), np.float32)
    for s, amp in ((6, 0.5), (12, 0.3), (24, 0.2)):
        small = rng.random((s * 9 // 16 + 2, s + 2)).astype(np.float32)
        im = Image.fromarray((small * 255).astype(np.uint8)).resize((W, H), Image.Resampling.BICUBIC)
        base += amp * np.asarray(im, np.float32) / 255
    base = (base - base.min()) / (base.max() - base.min() + 1e-6)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    head = (((xx - 0.66 * W) / (0.085 * W)) ** 2 + ((yy - 0.36 * H) / (0.17 * H)) ** 2) < 1
    body = (((xx - 0.66 * W) / (0.20 * W)) ** 2 + ((yy - 0.82 * H) / (0.34 * H)) ** 2) < 1
    img = np.stack([base * 90 + 14, base * 75 + 16, base * 95 + 26], axis=-1)
    img[head | body] *= 0.22
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


# --------------------------------------------------------------------------- #
# Görüntü işleme (numpy)
# --------------------------------------------------------------------------- #
def _process_base(img: Image.Image, palette: dict, focus: tuple[float, float], layout: str) -> Image.Image:
    img = ImageOps.autocontrast(img, cutoff=1)
    img = ImageEnhance.Contrast(img).enhance(1.25)
    img = ImageEnhance.Color(img).enhance(1.10)
    img = ImageEnhance.Brightness(img).enhance(0.92)
    arr = np.asarray(img, dtype=np.float32)

    # split-toning: gölgeler soğuk lacivert, vurgular palet tonu
    lum = arr.mean(axis=2, keepdims=True) / 255.0
    shadow = np.array([10, 16, 38], dtype=np.float32)
    tint = np.array(palette["tint"], dtype=np.float32)
    arr = arr * (1 - 0.20 * (1 - lum)) + shadow * 0.20 * (1 - lum)
    arr = arr + (tint - 128.0) * 0.10 * lum

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    fx, fy = focus[0] * W, focus[1] * H

    # odak noktasına spot ışık
    sigma = 0.28 * W
    spot = np.exp(-(((xx - fx) ** 2) + ((yy - fy) ** 2)) / (2 * sigma**2))[..., None]
    arr = arr + spot * (tint * 0.22)

    # vignette
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    vig = 1 - 0.55 * np.clip((r - 0.45) / 0.9, 0, 1) ** 1.6
    arr = arr * vig[..., None]

    # metin okunurluğu: alttan ve (sol yerleşimde) soldan gradyan
    ramp = np.clip((yy - H * 0.40) / (H * 0.60), 0, 1) ** 1.4
    dark = 0.78 * ramp
    if layout == "left":
        dark = np.maximum(dark, 0.55 * np.clip(1 - xx / (W * 0.62), 0, 1) ** 1.3)
    arr = arr * (1 - dark[..., None])

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGBA")


def _grain_and_sharpen(img: Image.Image, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    arr = arr + rng.normal(0, 4.5, (H, W, 1)).astype(np.float32)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return out.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=3))


# --------------------------------------------------------------------------- #
# Metin: otomatik sığdırma ve çizim
# --------------------------------------------------------------------------- #
def _partitions(words: list[str], n: int):
    for cuts in itertools.combinations(range(1, len(words)), n - 1):
        idx = (0, *cuts, len(words))
        yield [words[idx[i]: idx[i + 1]] for i in range(n)]


def _metrics(font: ImageFont.FreeTypeFont, sw: int, size: int):
    top = font.getbbox("H", anchor="ls")[1]
    cap_h = -top
    pitch = int(cap_h * 1.16 + sw)
    return cap_h, pitch


def fit_headline(text: str, max_w: int, max_h: int, font_path: Path | None, max_lines: int = 3):
    """En büyük puntoyu veren satır bölümünü seçer. (satırlar, punto, stroke, cap_h, pitch)"""
    words = text.split()
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    best = None
    for n in range(1, min(max_lines, len(words)) + 1):
        for part in _partitions(words, n):
            lines = [" ".join(p) for p in part]
            lo, hi, fit = 50, 360, None
            while lo <= hi:
                size = (lo + hi) // 2
                font = load_font(size, font_path)
                sw = max(6, size // 14)
                widest = max(
                    probe.textbbox((0, 0), ln, font=font, stroke_width=sw, anchor="ls")[2]
                    - probe.textbbox((0, 0), ln, font=font, stroke_width=sw, anchor="ls")[0]
                    for ln in lines
                )
                cap_h, pitch = _metrics(font, sw, size)
                height = cap_h + (n - 1) * pitch + 2 * sw
                if widest <= max_w and height <= max_h:
                    fit = (lines, size, sw, cap_h, pitch)
                    lo = size + 1
                else:
                    hi = size - 1
            if fit and (best is None or fit[1] > best[1] or (fit[1] == best[1] and len(fit[0]) < len(best[0]))):
                best = fit
    if best is None:  # çok uzun metin: tek satır, küçük punto
        font = load_font(50, font_path)
        best = ([text], 50, 6, *_metrics(font, 6, 50))
    return best


def _draw_block(draw, lines, font, sw, cap_h, pitch, x0, y0, align, block_w, colors, dx=0, dy=0, shadow=False):
    last_line = len(lines) - 1
    for i, line in enumerate(lines):
        base_y = y0 + cap_h + i * pitch + dy
        lw = draw.textlength(line, font=font)
        x = x0 + dx if align == "left" else x0 + (block_w - lw) / 2 + dx
        words = line.split(" ")
        for j, word in enumerate(words):
            is_accent = (i == last_line and j == len(words) - 1 and len(lines) + len(words) > 2)
            if shadow:
                fill, stroke = (0, 0, 0, 235), (0, 0, 0, 235)
            else:
                fill = colors["accent"] if (is_accent and colors["use_accent"]) else colors["fill"]
                stroke = (0, 0, 0, 255)
            draw.text((x, base_y), word, font=font, fill=fill, stroke_width=sw, stroke_fill=stroke, anchor="ls")
            x += draw.textlength(word + " ", font=font)


# --------------------------------------------------------------------------- #
# Ana API
# --------------------------------------------------------------------------- #
def build_thumbnail(
    bg: Image.Image | Path | str,
    headline: str,
    variant: int = 0,
    tag: str | None = None,
    out_path: Path | str | None = None,
    seed: int = 0,
) -> Path:
    palette = PALETTES[variant % len(PALETTES)]
    fx, fy, layout, w_ratio, h_ratio = LAYOUTS[variant % len(LAYOUTS)]
    font_path = find_font_path()

    img = Image.open(bg) if isinstance(bg, (str, Path)) else bg
    canvas = _process_base(_cover_resize(img, (fx, fy)), palette, (fx, fy), layout)

    text = sanitize_headline(headline)
    max_w, max_h = int(W * w_ratio), int(H * h_ratio)
    lines, size, sw, cap_h, pitch = fit_headline(text, max_w, max_h, font_path)
    font = load_font(size, font_path)

    probe = ImageDraw.Draw(canvas)
    block_w = max(int(probe.textlength(ln, font=font)) + 2 * sw for ln in lines)
    block_h = cap_h + (len(lines) - 1) * pitch + 2 * sw
    if layout == "center":
        x0 = (W - block_w) // 2
        y0 = H - block_h - 64
    else:
        x0 = 52
        y0 = H - block_h - 64 if variant % 3 == 0 else int((H - block_h) / 2) + 30

    colors = {"fill": palette["fill"] + (255,), "accent": palette["accent"] + (255,), "use_accent": variant % 2 == 0}

    # gölge katmanı (bulanık), sonra ana metin
    shadow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    _draw_block(ImageDraw.Draw(shadow_layer), lines, font, sw, cap_h, pitch, x0, y0, "left" if layout == "left" else "center",
                block_w, colors, dx=7, dy=10, shadow=True)
    canvas = Image.alpha_composite(canvas, shadow_layer.filter(ImageFilter.GaussianBlur(9)))
    draw = ImageDraw.Draw(canvas)

    # vurgu çizgisi (sol yerleşim)
    if layout == "left":
        bar_y = y0 + block_h + 10
        draw.rectangle([x0, bar_y, x0 + int(block_w * 0.55), bar_y + 12], fill=palette["accent"] + (255,))

    _draw_block(draw, lines, font, sw, cap_h, pitch, x0, y0, "left" if layout == "left" else "center", block_w, colors)

    # etiket bandı (üst sol)
    if tag:
        tfont = load_font(34, font_path)
        tag = tag.upper()
        tw = draw.textlength(tag, font=tfont)
        draw.rectangle([44, 40, 44 + tw + 36, 96], fill=palette["accent"] + (245,))
        draw.text((62, 68), tag, font=tfont, fill=(255, 255, 255, 255), anchor="lm")

    # marka damgası
    wfont = load_font(26, font_path)
    draw.text((W - 28, H - 22), config.CHANNEL_WATERMARK, font=wfont, fill=(255, 255, 255, 150), anchor="rs")

    final = _grain_and_sharpen(canvas, seed + variant)

    out_path = Path(out_path or (config.THUMB_DIR / f"thumb_v{variant + 1}.jpg"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for q in (93, 90, 86, 82, 76, 70):
        final.save(out_path, "JPEG", quality=q, optimize=True, progressive=True)
        if out_path.stat().st_size <= config.THUMB_MAX_BYTES:
            break
    log.info("Thumbnail yazıldı: %s (%d KB, varyant %d)", out_path.name, out_path.stat().st_size // 1024, variant + 1)
    return out_path


def build_thumbnails(
    bg: Image.Image | Path | str,
    headline: str,
    tag: str | None = None,
    out_dir: Path | str | None = None,
    variants: int | None = None,
    seed: int = 0,
) -> list[Path]:
    out_dir = Path(out_dir or config.THUMB_DIR)
    n = variants or config.THUMB_VARIANTS
    return [build_thumbnail(bg, headline, v, tag, out_dir / f"thumb_v{v + 1}.jpg", seed) for v in range(n)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Thumbnail demo / manuel üretim")
    ap.add_argument("--text", default="VERBOTENE AKTE")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--bg", default=None, help="arka plan görseli (yoksa prosedürel demo)")
    ap.add_argument("--out", default=str(config.THUMB_DIR))
    ap.add_argument("--variants", type=int, default=3)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    background = Image.open(a.bg) if a.bg else procedural_background(7)
    for p in build_thumbnails(background, a.text, a.tag, a.out, a.variants):
        print(p)
