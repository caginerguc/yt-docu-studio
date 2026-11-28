"""Vor der Realität — kanal görselleri: banner (2560x1440) ve profil resmi (800x800).

Karanlık arşiv teması: raflarda dosya kutuları, spot ışığı, sis, film greni; kırmızı arşiv damgası vurgusu.
Çıktılar bu dosyanın klasörüne yazılır.
"""
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).parent
RNG = np.random.default_rng(1981)

PAPER = (232, 220, 194)
RED = (179, 20, 27)
SERIF_BOLD = "/usr/share/fonts/truetype/crosextra/Caladea-Bold.ttf"
MONO_BOLD = "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf"


def font(path, size):
    return ImageFont.truetype(path, size)


def radial(w, h, cx, cy, rx, ry):
    """0..1 merkezde 1 olan elips gradyanı."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2)
    return np.clip(1 - d, 0, 1)


def grain(img, amount=9):
    a = np.asarray(img).astype(np.int16)
    n = RNG.normal(0, amount, a.shape[:2])[..., None]
    return Image.fromarray(np.clip(a + n, 0, 255).astype(np.uint8))


def spaced(draw, xy, text, fnt, fill, spacing, anchor_center=True):
    """Harf aralıklı metin; xy = merkez (yatay) ve taban çizgisi."""
    widths = [draw.textlength(ch, font=fnt) for ch in text]
    total = sum(widths) + spacing * (len(text) - 1)
    x = xy[0] - total / 2 if anchor_center else xy[0]
    for ch, w in zip(text, widths):
        draw.text((x, xy[1]), ch, font=fnt, fill=fill, anchor="ls")
        x += w + spacing
    return total


# --------------------------------------------------------------------------- #
# Arşiv rafları
# --------------------------------------------------------------------------- #
def shelves(w, h, rows, top, row_h, seed):
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(img)
    for r in range(rows):
        y_floor = top + (r + 1) * row_h
        x = rng.integers(-40, 0)
        while x < w:
            bw = int(rng.choice([46, 58, 70, 90, 120, 150]) * rng.uniform(0.85, 1.15))
            bh = int(row_h * rng.uniform(0.55, 0.9))
            base = rng.uniform(18, 38)
            tint = rng.choice([(1.15, 1.0, 0.82), (1.0, 1.0, 1.0), (1.1, 0.95, 0.85), (0.9, 0.92, 1.0)])
            col = tuple(int(base * t) for t in tint)
            if rng.random() < 0.12:                         # boşluk
                x += bw
                continue
            d.rectangle([x, y_floor - bh, x + bw - 3, y_floor - 6], fill=col)
            # kutu üstü kenar ışığı
            d.line([x, y_floor - bh, x + bw - 3, y_floor - bh], fill=tuple(min(255, c + 14) for c in col), width=2)
            # etiket
            if bw > 52 and rng.random() < 0.8:
                lw, lh = int(bw * 0.5), int(min(bh * 0.22, 42))
                lx, ly = x + (bw - lw) // 2, y_floor - bh + int(bh * 0.25)
                lab = tuple(int(c * rng.uniform(0.22, 0.32)) for c in PAPER)
                d.rectangle([lx, ly, lx + lw, ly + lh], fill=lab)
                if rng.random() < 0.18:                     # kırmızı işaret
                    d.rectangle([lx, ly + lh + 6, lx + lw // 3, ly + lh + 12], fill=(70, 12, 14))
            x += bw
        d.rectangle([0, y_floor - 6, w, y_floor + 10], fill=(24, 20, 17))   # raf tahtası
        d.line([0, y_floor - 6, w, y_floor - 6], fill=(44, 37, 30), width=2)
    return img


def fog(w, h, seed, scale=24, blur=60):
    rng = np.random.default_rng(seed)
    small = rng.random((h // scale + 2, w // scale + 2)).astype(np.float32)
    im = Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    return im.filter(ImageFilter.GaussianBlur(blur))


# --------------------------------------------------------------------------- #
# Banner
# --------------------------------------------------------------------------- #
def banner():
    W, H = 2560, 1440
    SAFE = (507, 508, 2053, 931)                         # tüm cihazlarda görünen alan (1546x423)
    cx, cy = W / 2, (SAFE[1] + SAFE[3]) / 2

    bg = Image.new("RGB", (W, H), (9, 9, 11))
    sh = shelves(W, H, rows=6, top=40, row_h=230, seed=7).filter(ImageFilter.GaussianBlur(2.2))
    # raflar: merkezde hafif görünür, kenarlara doğru kaybolur
    light = radial(W, H, cx, cy + 40, 1500, 760)
    sh_a = np.asarray(sh).astype(np.float32) * (0.28 + 0.95 * light[..., None])
    img = np.asarray(bg).astype(np.float32) + sh_a

    # sıcak spot ışığı ve sis
    glow = radial(W, H, cx, cy - 20, 900, 420) ** 1.6
    img += glow[..., None] * np.array([70, 52, 30], np.float32)
    fg = np.asarray(fog(W, H, 3)).astype(np.float32) / 255
    img += (fg ** 2.2 * 46 * (0.4 + light))[..., None] * np.array([1.0, 0.95, 0.88], np.float32)

    # metin bölgesini karart (okunabilirlik)
    band = radial(W, H, cx, cy, 1050, 250) ** 0.9
    img *= (1 - 0.55 * band)[..., None]

    # vinyet
    vig = radial(W, H, cx, H / 2, W * 0.75, H * 0.85)
    img *= (0.25 + 0.75 * vig ** 0.8)[..., None]
    out = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

    # ---- metin
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    title_f = font(SERIF_BOLD, 138)
    mono_f = font(MONO_BOLD, 40)
    tag_f = font(MONO_BOLD, 30)

    title = "VOR DER REALITÄT"
    base_y = cy + 40
    # gölge
    sh_l = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    spaced(ImageDraw.Draw(sh_l), (cx + 6, base_y + 8), title, title_f, (0, 0, 0, 230), 10)
    layer = Image.alpha_composite(layer, sh_l.filter(ImageFilter.GaussianBlur(10)))
    d = ImageDraw.Draw(layer)
    tw = spaced(d, (cx, base_y), title, title_f, PAPER + (255,), 10)

    assert tw < (SAFE[2] - SAFE[0]) - 120, f"başlık güvenli alana sığmıyor: {tw:.0f}px"
    # üst etiket: kırmızı damga bandı
    tag = "AKTENARCHIV"
    tag_w = spaced(ImageDraw.Draw(Image.new("RGBA", (10, 10))), (0, 0), tag, tag_f, (0, 0, 0, 0), 8, False)
    ty = base_y - 138 - 30
    d.rectangle([cx - tag_w / 2 - 22, ty - 40, cx + tag_w / 2 + 22, ty + 12], fill=RED + (235,))
    spaced(d, (cx, ty), tag, tag_f, (255, 245, 235, 255), 8)

    # ince kırmızı çizgi + alt slogan
    ly = base_y + 34
    d.rectangle([cx - tw * 0.32, ly, cx + tw * 0.32, ly + 5], fill=RED + (255,))
    spaced(d, (cx, ly + 76), "GESCHICHTE · SKANDALE · UNGELÖSTE AKTEN", mono_f, PAPER + (215,), 6)

    out = Image.alpha_composite(out.convert("RGBA"), layer).convert("RGB")
    out = grain(out, 8)
    out.save(OUT / "banner_2560x1440.jpg", quality=92, optimize=True)   # YouTube sınırı 6 MB
    return out, SAFE


# --------------------------------------------------------------------------- #
# Profil resmi
# --------------------------------------------------------------------------- #
def ring_text(d, cx, cy, r, text, fnt, fill, start_deg=-90):
    circ = 2 * math.pi * r
    widths = [d.textlength(ch, font=fnt) for ch in text]
    total = sum(widths)
    gap = (circ - total) / len(text)
    ang = math.radians(start_deg)
    for ch, w in zip(text, widths):
        step = (w + gap) / r
        a = ang + step / 2
        tile = Image.new("RGBA", (int(w) + 40, fnt.size + 40), (0, 0, 0, 0))
        ImageDraw.Draw(tile).text((tile.width / 2, tile.height / 2), ch, font=fnt, fill=fill, anchor="mm")
        tile = tile.rotate(-math.degrees(a) - 90, resample=Image.BICUBIC, expand=True)
        x, y = cx + r * math.cos(a), cy + r * math.sin(a)
        yield tile, (int(x - tile.width / 2), int(y - tile.height / 2))
        ang += step


def profile():
    S = 800
    c = S / 2
    img = np.zeros((S, S, 3), np.float32) + np.array([10, 10, 12], np.float32)
    glow = radial(S, S, c, c - 30, 420, 420) ** 1.4
    img += glow[..., None] * np.array([62, 46, 28], np.float32)
    fg = np.asarray(fog(S, S, 11, scale=16, blur=28)).astype(np.float32) / 255
    img += (fg ** 2 * 26)[..., None]
    base = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert("RGBA")

    # damga katmanı (kırmızı halka + çevre yazısı)
    stamp = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(stamp)
    R_out, R_in = 330, 238
    d.ellipse([c - R_out, c - R_out, c + R_out, c + R_out], outline=RED + (255,), width=26)
    d.ellipse([c - R_in, c - R_in, c + R_in, c + R_in], outline=RED + (255,), width=10)
    rf = font(MONO_BOLD, 50)
    for tile, pos in ring_text(d, c, c, (R_out + R_in) / 2 + 4, "VOR DER REALITÄT • AKTENARCHIV • ", rf,
                               RED + (255,)):
        stamp.alpha_composite(tile, pos)
    # aşınmış mürekkep dokusu
    a = np.asarray(stamp.split()[3]).astype(np.float32)
    worn = np.asarray(fog(S, S, 5, scale=3, blur=0.7)).astype(np.float32) / 255
    a *= np.clip(0.55 + (worn - 0.3) * 1.6, 0.6, 1.0)
    stamp.putalpha(Image.fromarray(a.astype(np.uint8)))
    stamp = stamp.rotate(-8, resample=Image.BICUBIC, center=(c, c))

    # anahtar deliği (kağıt rengi, hafif gölgeli)
    key = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    k = ImageDraw.Draw(key)
    hr = 74
    hy = c - 52
    k.ellipse([c - hr, hy - hr, c + hr, hy + hr], fill=PAPER + (255,))
    k.polygon([(c - 46, hy + 30), (c + 46, hy + 30), (c + 78, c + 150), (c - 78, c + 150)], fill=PAPER + (255,))
    shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    shadow.paste((0, 0, 0, 200), (0, 0), key.split()[3])
    shadow = ImageChops.offset(shadow, 6, 10).filter(ImageFilter.GaussianBlur(12))

    out = Image.alpha_composite(base, stamp)
    out = Image.alpha_composite(out, shadow)
    out = Image.alpha_composite(out, key).convert("RGB")
    out = grain(out, 7)
    out.save(OUT / "profil_800x800.png", optimize=True)
    return out


# --------------------------------------------------------------------------- #
# Önizlemeler: güvenli alanlar ve küçük boyut
# --------------------------------------------------------------------------- #
def previews(bn, safe, pf):
    W, H = bn.size
    p = bn.copy().convert("RGBA")
    o = Image.new("RGBA", p.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(o)
    d.rectangle([0, (H - 423) // 2, W - 1, (H + 423) // 2], outline=(80, 200, 255, 255), width=6)   # masaüstü
    d.rectangle(safe, outline=(255, 210, 0, 255), width=6)                                          # telefon
    p = Image.alpha_composite(p, o).convert("RGB").resize((1280, 720), Image.LANCZOS)
    p.save(OUT / "onizleme_banner_guvenli_alanlar.jpg", quality=88)

    desk = bn.crop((0, (H - 423) // 2, W, (H + 423) // 2)).resize((1280, 212), Image.LANCZOS)
    mob = bn.crop(safe).resize((773, 212), Image.LANCZOS)
    sheet = Image.new("RGB", (1280, 212 * 2 + 30), (30, 30, 30))
    sheet.paste(desk, (0, 0))
    sheet.paste(mob, ((1280 - 773) // 2, 212 + 30))
    sheet.save(OUT / "onizleme_banner_masaustu_telefon.jpg", quality=88)

    sizes = [800, 176, 98, 48]
    sheet = Image.new("RGB", (sum(sizes) + 40 * (len(sizes) + 1), 840), (245, 245, 245))
    x = 40
    for s in sizes:
        im = pf.resize((s, s), Image.LANCZOS)
        m = Image.new("L", (s, s), 0)
        ImageDraw.Draw(m).ellipse([0, 0, s - 1, s - 1], fill=255)
        sheet.paste(im, (x, 20), m)
        x += s + 40
    sheet.save(OUT / "onizleme_profil_boyutlar.jpg", quality=90)


if __name__ == "__main__":
    bn, safe = banner()
    pf = profile()
    previews(bn, safe, pf)
    print("tamam")
