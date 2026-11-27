"""Aşama 5b — YouTube Short (dikey fragman, 1080x1920).

Belgeselin seçilen cümleleri, belgeselin kendi görüntüsü ve anlatımıyla yeniden kurgulanır (yeni ses/görsel
üretimi yok, yeni bilgi hatası riski yok). Girdi, VideoRenderer'ın render() sonrası bıraktığı ara dosyalardır:
  work_dir/video_only.mp4, work_dir/narration.wav ve cümle zamanları (renderer.timings).

Görünüm
  * arkada aynı görüntünün bulanık, karartılmış hali; ortada kenarları kırpılmış ön plan
  * üstte kanal etiketi + ilk saniyelerde kanca yazısı
  * büyük, 3-4 kelimelik yakılmış altyazılar (Shorts çoğu zaman sessiz izlenir)
  * sonda kapanış kartı: "Die ganze Doku: ▶ auf dem Kanal" + belgeselin adı
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from src import config
from src.plan import ProductionPlan, ShortPlan
from src.video_renderer import (FPS, MUSIC_LUFS, SR, VideoRenderer, X264, _run, probe_duration,
                                split_words)

log = logging.getLogger("makrokontext.short")

SW, SH = config.SHORTS_W, config.SHORTS_H
END_CARD_S = 2.6
HOOK_S = 3.2                    # kanca yazısının ekranda kalma süresi
FG_H, FG_Y = 1000, 460          # ön plan yüksekliği ve üst kenarı (üstü yazı, altı Shorts arayüzü için boş)
CAP_WORDS = 4                   # altyazı parçası başına en fazla kelime
CUT_TIMEOUT_S = 180             # tek kesit/birleştirme için FFmpeg süre sınırı
COMPOSE_TIMEOUT_S = 900         # dikey kurgu için FFmpeg süre sınırı (takılırsa Short atlanır, iş sürer)
LEAD_S, TAIL_S = 0.12, 0.25     # cümle sesinin öncesinden/sonrasından alınan pay
FONT = "DejaVu Sans"            # Actions'ta fonts-dejavu-core ile kurulu; ▶ ve Ä/Ö/Ü/ß içerir

_END_TEXT = {
    "de": ("Die ganze Doku", "▶ auf dem Kanal"),
    "en": ("Full documentary", "▶ on the channel"),
    "tr": ("Belgeselin tamamı", "▶ kanalda"),
}


@dataclass
class Segment:
    sid: str
    text: str
    start: float          # belgesel zaman çizelgesinde
    end: float
    speech_from: float    # Short zaman çizelgesinde konuşmanın başı/sonu
    speech_to: float


def _ass_time(t: float) -> str:
    cs = max(0, int(round(t * 100)))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6_000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " ")


def _wrap(text: str, width: int) -> str:
    """Kelime bazında satır kırma (ASS \\N)."""
    lines, cur = [], ""
    for w in text.split():
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\\N".join(lines)


def _bgr(rgb: tuple[int, int, int], alpha: int = 0) -> str:
    """Stil rengi: &HAABBGGRR"""
    r, g, b = rgb
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


def _inline_color(rgb: tuple[int, int, int]) -> str:
    """Satır içi renk: {\\c&HBBGGRR&}"""
    r, g, b = rgb
    return f"{{\\c&H{b:02X}{g:02X}{r:02X}&}}"


class ShortRenderer:
    def __init__(self, plan: ProductionPlan, short: ShortPlan, renderer: VideoRenderer):
        self.plan, self.short, self.r = plan, short, renderer
        self.work = renderer.work_dir
        self.out = config.OUTPUT_DIR / "short.mp4"
        self.lang = plan.language.lower()

    # ------------------------------------------------------------------ #
    def segments(self) -> list[Segment]:
        by_id = {tm.sentence.id: tm for tm in self.r.timings}
        picked = [by_id[i] for i in self.short.sentence_ids if i in by_id]
        missing = [i for i in self.short.sentence_ids if i not in by_id]
        if missing:
            log.warning("Short: plandaki %s cümleleri belgeselde yok, atlandı.", missing)
        limit = config.SHORTS_MAX_SECONDS - END_CARD_S - 0.5

        def total(tms) -> float:
            return sum(min(tm.end + TAIL_S, self.r.duration) - max(tm.start, tm.speech_start - LEAD_S) for tm in tms)

        while len(picked) > 2 and total(picked) > limit:      # baş (kanca) ve son (soru) kalır, ortadan çıkar
            picked.pop(len(picked) // 2)
        if picked and total(picked) > limit:
            picked = picked[:1]
        segs, t = [], 0.0
        for tm in picked:
            a = max(tm.start, tm.speech_start - LEAD_S)
            b = min(tm.end + TAIL_S, self.r.duration)
            segs.append(Segment(tm.sentence.id, tm.sentence.text, a, b, t + (tm.speech_start - a), t + (tm.end - a)))
            t += b - a
        return segs

    # ------------------------------------------------------------------ #
    def _cut(self, segs: list[Segment]) -> tuple[Path, Path]:
        """Her cümleyi ayrı, basit bir FFmpeg çağrısıyla keser (girişte arama) ve parçaları kopyalayarak birleştirir.
        Aynı girdiyi tek komutta birden çok trim dalıyla okumak FFmpeg 7+ zamanlayıcısında kilitlenebiliyor
        (cümleler belgesel sırasında değilse); bu yüzden kesitler tek tek alınır."""
        video_src = self.work / "video_only.mp4"
        narr_src = self.work / "narration.wav"
        for p in (video_src, narr_src):
            if not p.exists():
                raise FileNotFoundError(f"Belgesel ara dosyası yok: {p.name} (önce belgesel render edilmeli)")
        vparts, aparts = [], []
        for i, s in enumerate(segs):
            d = s.end - s.start
            v, a = self.work / f"short_seg{i}.mp4", self.work / f"short_seg{i}.wav"
            _run(["ffmpeg", "-y", "-ss", f"{s.start:.3f}", "-i", str(video_src), "-frames:v",
                  str(max(1, round(d * FPS))), "-an", *X264, str(v)], f"short kesit {i + 1} (görüntü)",
                 timeout=CUT_TIMEOUT_S)
            _run(["ffmpeg", "-y", "-ss", f"{s.start:.3f}", "-i", str(narr_src), "-t", f"{d:.3f}",
                  "-af", f"afade=t=in:d=0.03,afade=t=out:st={max(0.0, d - 0.06):.3f}:d=0.06",
                  "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", str(a)], f"short kesit {i + 1} (ses)",
                 timeout=CUT_TIMEOUT_S)
            vparts.append(v)
            aparts.append(a)
        v_out, a_out = self.work / "short_h.mp4", self.work / "short_narr.wav"
        for parts, out, what in ((vparts, v_out, "görüntü"), (aparts, a_out, "ses")):
            lst = self.work / f"short_{what}.txt"
            lst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in parts), encoding="utf-8")
            _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)],
                 f"short birleştirme ({what})", timeout=CUT_TIMEOUT_S)
        return v_out, a_out

    # ------------------------------------------------------------------ #
    def _ass(self, segs: list[Segment], total: float) -> Path:
        paper, red = (232, 220, 194), (179, 20, 27)
        head, sub = _END_TEXT.get(self.lang, _END_TEXT["en"])
        title = self.plan.seo.title if len(self.plan.seo.title) <= 70 else self.plan.seo.title[:67].rsplit(" ", 1)[0] + "…"
        styles = [
            # Ad, Font, Boyut, Ana renk, Kontur rengi, Kalın, Kontur, Gölge, Hizalama, Kenar L/R/V, Kutu stili
            ("Brand", 34, _bgr((255, 255, 255)), _bgr(red), 1, 10, 0, 8, 60, 120, 3),
            ("Hook", 66, _bgr(paper), _bgr((0, 0, 0)), 1, 5, 2, 8, 70, 215, 1),
            ("Cap", 72, _bgr((255, 255, 255)), _bgr((0, 0, 0)), 1, 6, 2, 2, 60, 470, 1),
            ("EndHead", 82, _bgr(paper), _bgr((0, 0, 0)), 1, 5, 0, 5, 60, 0, 1),
            ("EndTitle", 46, _bgr((255, 255, 255)), _bgr((0, 0, 0)), 0, 4, 0, 2, 80, 560, 1),
        ]
        lines = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {SW}", f"PlayResY: {SH}", "WrapStyle: 0",
                 "ScaledBorderAndShadow: yes", "", "[V4+ Styles]",
                 "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
                 "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
                 "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"]
        for name, size, prim, outl, bold, ol, sh, align, mlr, mv, border in styles:
            lines.append(f"Style: {name},{FONT},{size},{prim},{prim},{outl},&H80000000,{-1 if bold else 0},0,0,0,"
                         f"100,100,1,0,{border},{ol},{sh},{align},{mlr},{mlr},{mv},1")
        lines += ["", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]

        def ev(start: float, end: float, style: str, text: str, layer: int = 0) -> None:
            lines.append(f"Dialogue: {layer},{_ass_time(start)},{_ass_time(end)},{style},,0,0,0,,{text}")

        end_at = total - END_CARD_S
        ev(0, end_at, "Brand", _ass_escape(config.CHANNEL_NAME.upper()))
        ev(0.15, min(HOOK_S, end_at), "Hook", "{\\fad(150,250)}" + _wrap(_ass_escape(self.short.hook_text), 24))
        for s in segs:
            words = s.text.split()
            chunks = split_words(words, max(1, -(-len(words) // CAP_WORDS)), slack=1)
            span = max(0.4, s.speech_to - s.speech_from)
            weights = [len(" ".join(c)) for c in chunks]
            t = s.speech_from
            for c, w in zip(chunks, weights):
                d = span * w / (sum(weights) or 1)
                ev(t, t + d - 0.02, "Cap", _ass_escape(" ".join(c)), layer=1)
                t += d
        ev(end_at, total, "EndHead", "{\\fad(200,0)}" + _ass_escape(head) + "\\N" + _inline_color(red)
           + _ass_escape(sub), layer=2)
        ev(end_at, total, "EndTitle", "{\\fad(300,0)}„" + _wrap(_ass_escape(title), 28) + "“", layer=2)
        path = self.work / "short.ass"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    # ------------------------------------------------------------------ #
    def _compose(self, v_in: Path, a_in: Path, ass: Path, total: float) -> None:
        end_at = total - END_CARD_S
        music = self.r.music_paths[0] if self.r.music_paths else None
        inputs = ["-i", str(v_in), "-i", str(a_in)]
        if music:
            inputs += ["-stream_loop", "-1", "-ss", "12", "-i", str(music)]
        fc = [
            f"[0:v]tpad=stop_mode=clone:stop_duration={END_CARD_S},split=2[b][f]",
            f"[b]scale=-2:{SH},crop={SW}:{SH},gblur=sigma=40,eq=brightness=-0.12:saturation=0.7[bg]",
            f"[f]scale=-2:{FG_H},crop={SW}:{FG_H}[fg]",
            f"[bg][fg]overlay=0:{FG_Y},drawbox=x=0:y=0:w=iw:h=ih:color=black@0.62:t=fill:"
            f"enable='gte(t,{end_at:.3f})',subtitles=filename={ass.name},format=yuv420p[v]",
        ]
        norm = f"aresample={SR},aformat=sample_fmts=fltp:channel_layouts=stereo"
        if music:
            fc += [f"[1:a]apad=pad_dur={END_CARD_S},{norm},asplit=2[voice][key]",
                   f"[2:a]{norm},atrim=0:{total:.3f},loudnorm=I={MUSIC_LUFS + 2}:TP=-8:LRA=14,aresample={SR},"
                   f"afade=t=in:d=0.6,afade=t=out:st={max(0.0, total - 1.6):.3f}:d=1.6[bed]",
                   "[bed][key]sidechaincompress=threshold=0.015:ratio=6:attack=40:release=700:makeup=1[duck]",
                   "[voice][duck]amix=inputs=2:duration=first:normalize=0,loudnorm=I=-14:TP=-1.5:LRA=11[a]"]
        else:
            fc.append(f"[1:a]apad=pad_dur={END_CARD_S},{norm},loudnorm=I=-14:TP=-1.5:LRA=11[a]")
        tmp = self.out.with_suffix(".part.mp4")
        _run(["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(fc), "-map", "[v]", "-map", "[a]",
              "-t", f"{total:.3f}", *X264, "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
              str(tmp.resolve())], "short kurgu", cwd=self.work, timeout=COMPOSE_TIMEOUT_S)
        tmp.replace(self.out)

    # ------------------------------------------------------------------ #
    def render(self) -> Path | None:
        segs = self.segments()
        if not segs:
            log.warning("Short: seçilecek cümle bulunamadı, Short üretilmedi.")
            return None
        v_in, a_in = self._cut(segs)
        total = probe_duration(a_in) + END_CARD_S
        ass = self._ass(segs, total)
        self._compose(v_in, a_in, ass, total)
        real = probe_duration(self.out)
        meta = {"duration_s": round(real, 2), "sentences": [s.sid for s in segs], "hook_text": self.short.hook_text,
                "title": self.short.title, "fps": FPS, "size": f"{SW}x{SH}"}
        (config.OUTPUT_DIR / "short.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                      encoding="utf-8")
        log.info("Short hazır: %s (%.1f sn, %d cümle)", self.out.name, real, len(segs))
        return self.out


def render_short(plan: ProductionPlan, renderer: VideoRenderer) -> tuple[Path | None, ShortPlan]:
    short = plan.short
    if short is None or not short.sentence_ids:      # eski planlar veya seçim yapılamadıysa
        short = ShortPlan.fallback(plan.acts, plan.seo)
    return ShortRenderer(plan, short, renderer).render(), short
