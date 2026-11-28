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
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import config
from src.plan import ProductionPlan, ShortPlan
from src.video_renderer import FPS, SR, RenderError, VideoRenderer, X264, _run, probe_duration, split_words

log = logging.getLogger("makrokontext.short")

SW, SH = config.SHORTS_W, config.SHORTS_H
END_CARD_S = 2.6
HOOK_S = 3.2                    # kanca yazısının ekranda kalma süresi
FG_H, FG_Y = 1000, 460          # ön plan yüksekliği ve üst kenarı (üstü yazı, altı Shorts arayüzü için boş)
CAP_WORDS = 4                   # altyazı parçası başına en fazla kelime
CUT_TIMEOUT_S = 180             # tek kesit/birleştirme için FFmpeg süre sınırı
COMPOSE_TIMEOUT_S = 900         # dikey kurgu için FFmpeg süre sınırı (takılırsa Short atlanır, iş sürer)
LEAD_S, TAIL_S = 0.12, 0.25     # cümle sesinin öncesinden/sonrasından alınan pay
MUSIC_OFFSET_S = 12.0           # müziğe sessiz girişi atlayarak başla
MAX_SILENCE_S = 1.2             # ses kontrolü: kapanış kartından önce bundan uzun tam sessizlik = hata
SILENCE_DB = -55.0
NARR_MAX_GAP_S = 1.5            # bir cümlenin konuşma bölgesi içinde bundan uzun sessizlik = kesik anlatım
MUSIC_BELOW_VOICE_DB = 12.0     # müzik yatağı, konuşma seviyesinin bu kadar altında (duraklamalarda)
DUCK_DB = -10.0                 # anlatım sürerken müzik ayrıca bu kadar kısılır
DUCK_ATTACK_S, DUCK_RELEASE_S, DUCK_HOLD_S = 0.04, 0.7, 0.25
FRAME_S = 0.01                  # konuşma algılama penceresi
SPEECH_DB = -40.0               # bu seviyenin üstü konuşma sayılır
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


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    """16 bit WAV -> (örnek x kanal) float32 [-1, 1] ve örnekleme hızı."""
    with wave.open(str(path), "rb") as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        if width != 2:
            raise RenderError(f"Beklenmeyen ses biçimi ({path.name}): {width * 8} bit")
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, ch)
    return x.astype(np.float32) / 32768, sr


def _write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    x = x if x.ndim == 2 else x[:, None]
    pcm = np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(x.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt((x.astype(np.float32) ** 2).mean())) if x.size else 0.0


def _db(x: np.ndarray) -> float:
    return 20 * np.log10(_rms(x) + 1e-9) if x.size else -120.0


def _longest_silence(x: np.ndarray, sr: int, t0: float, t1: float, win_s: float = 0.05) -> tuple[float, float]:
    """[t0, t1] aralığındaki en uzun tam sessizlik (50 ms pencerelerle, SILENCE_DB altı): (süre, başladığı an)."""
    a, b = max(0, int(t0 * sr)), min(len(x), int(t1 * sr))
    win = max(1, int(win_s * sr))
    n = (b - a) // win
    if n <= 0:
        return 0.0, t0
    db = 20 * np.log10(np.sqrt((x[a:a + n * win].reshape(n, win) ** 2).mean(1)) + 1e-9)
    best, best_at, run, start = 0, 0, 0, 0
    for i, quiet in enumerate(db < SILENCE_DB):
        if quiet:
            if run == 0:
                start = i
            run += 1
            if run > best:
                best, best_at = run, start
        else:
            run = 0
    return best * win_s, a / sr + best_at * win_s


def _speech_frames(v: np.ndarray) -> np.ndarray:
    """10 ms pencerelerde konuşma var mı (kısa kelime araları da konuşmaya dahil edilir)."""
    hop = int(SR * FRAME_S)
    n = len(v) // hop
    if n == 0:
        return np.zeros(0, bool)
    db = 20 * np.log10(np.sqrt((v[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)
    hold = max(1, int(DUCK_HOLD_S / FRAME_S))
    return np.convolve((db > SPEECH_DB).astype(np.float32), np.ones(hold, np.float32), mode="same") > 0


def _duck_gain(active: np.ndarray, n: int) -> np.ndarray:
    """Konuşma sırasında müziği DUCK_DB kadar kısan, yumuşak (atak/bırakma) kazanç eğrisi; örnek başına."""
    if not len(active):
        return np.ones(n, np.float32)
    target = np.where(active, 10 ** (DUCK_DB / 20), 1.0)
    att, rel = 1 - np.exp(-FRAME_S / DUCK_ATTACK_S), 1 - np.exp(-FRAME_S / DUCK_RELEASE_S)
    g, cur = np.empty(len(target), np.float32), 1.0
    for i, tg in enumerate(target):
        cur += (tg - cur) * (att if tg < cur else rel)
        g[i] = cur
    hop = int(SR * FRAME_S)
    centers = (np.arange(len(g)) + 0.5) * hop
    return np.interp(np.arange(n, dtype=np.float32), centers, g).astype(np.float32)


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
            b = a + max(1, round((b - a) * FPS)) / FPS          # kare ızgarası: görüntü = ses uzunluğu
            segs.append(Segment(tm.sentence.id, tm.sentence.text, a, b, t + (tm.speech_start - a), t + (tm.end - a)))
            t += b - a
        return segs

    # ------------------------------------------------------------------ #
    def _cut(self, segs: list[Segment]) -> tuple[Path, Path]:
        """Görüntü: her cümle ayrı FFmpeg çağrısıyla (girişte arama) kesilip kopyalanarak birleştirilir.
        Ses: anlatım WAV'ı Python'da örnek hassasiyetinde kesilir. (FFmpeg 8'de WAV'da '-ss' + afade, bazı
        kesitlerin ilk saniyelerini sıfırladı; tek komutta ortak girdi ise FFmpeg 7+'da kilitlenebiliyor.)"""
        video_src = self.work / "video_only.mp4"
        narr_src = self.work / "narration.wav"
        for p in (video_src, narr_src):
            if not p.exists():
                raise FileNotFoundError(f"Belgesel ara dosyası yok: {p.name} (önce belgesel render edilmeli)")
        vparts = []
        for i, s in enumerate(segs):
            v = self.work / f"short_seg{i}.mp4"
            _run(["ffmpeg", "-y", "-ss", f"{s.start:.3f}", "-i", str(video_src), "-frames:v",
                  str(max(1, round((s.end - s.start) * FPS))), "-an", *X264, str(v)],
                 f"short kesit {i + 1} (görüntü)", timeout=CUT_TIMEOUT_S)
            vparts.append(v)
        v_out, a_out = self.work / "short_h.mp4", self.work / "short_narr.wav"
        lst = self.work / "short_video.txt"
        lst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in vparts), encoding="utf-8")
        _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(v_out)],
             "short birleştirme (görüntü)", timeout=CUT_TIMEOUT_S)
        self._cut_audio(narr_src, segs, a_out)
        return v_out, a_out

    @staticmethod
    def _cut_audio(src: Path, segs: list[Segment], out: Path) -> None:
        """Anlatımı örnek (sample) hassasiyetinde keser; her kesite 30/60 ms yumuşak giriş/çıkış."""
        with wave.open(str(src), "rb") as w:
            sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            if width != 2:
                raise RenderError(f"Beklenmeyen anlatım biçimi: {width * 8} bit")
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, ch)
        parts = []
        for s in segs:
            a = int(round(s.start * sr))
            n = int(round((s.end - s.start) * sr))
            x = data[a:a + n].astype(np.float32)
            if len(x) < n:                                   # belgesel sonunu aşarsa sessizlikle tamamla
                x = np.vstack([x, np.zeros((n - len(x), ch), np.float32)])
            fi, fo = min(len(x), int(0.03 * sr)), min(len(x), int(0.06 * sr))
            if fi:
                x[:fi] *= np.linspace(0, 1, fi, dtype=np.float32)[:, None]
            if fo:
                x[-fo:] *= np.linspace(1, 0, fo, dtype=np.float32)[:, None]
            parts.append(x)
        pcm = np.clip(np.vstack(parts), -32768, 32767).astype(np.int16)
        with wave.open(str(out), "wb") as w:
            w.setnchannels(ch)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())

    def _music_bed(self, total: float) -> Path | None:
        """Müzik yatağını ayrı ve basit bir adımda hazırlar (döngü ayarı yok; FFmpeg 8'de -stream_loop + -ss
        müziği tamamen düşürmüştü)."""
        if not self.r.music_paths:
            return None
        src = self.r.music_paths[0]
        dur = probe_duration(src)
        offset = MUSIC_OFFSET_S if dur > total + MUSIC_OFFSET_S + 1 else max(0.0, dur - total - 1)
        bed = self.work / "short_music.wav"
        _run(["ffmpeg", "-y", "-ss", f"{offset:.2f}", "-i", str(src), "-t", f"{total + 0.5:.3f}", "-vn",
              "-ac", "2", "-ar", str(SR), "-c:a", "pcm_s16le", str(bed)], "short müzik", timeout=CUT_TIMEOUT_S)
        return bed

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
    def _mix(self, narr: Path, music: Path | None, total: float) -> tuple[Path, Path]:
        """Ses tamamen Python'da karıştırılır: anlatım + müzik yatağı + anlatım sırasında müziği kısma + geçişler.
        (FFmpeg 8'de apad/amix/sidechaincompress zinciri Short'ta iki kez sessiz delikler bıraktı.)
        Dönüş: (ham karışım, ses yüksekliği ayarlanmış son ses)."""
        voice, sr = _read_wav(narr)
        if sr != SR:
            raise RenderError(f"Beklenmeyen anlatım örnekleme hızı: {sr}")
        n = int(round(total * SR))
        v = np.zeros(n, np.float32)
        k = min(n, len(voice))
        v[:k] = voice[:k].mean(axis=1)
        mix = np.repeat(v[:, None], 2, axis=1)
        if music is not None:
            mus, msr = _read_wav(music)
            if msr != SR:
                raise RenderError(f"Beklenmeyen müzik örnekleme hızı: {msr}")
            bed = np.zeros((n, 2), np.float32)
            k = min(n, len(mus))
            bed[:k] = mus[:k, :2] if mus.shape[1] >= 2 else np.repeat(mus[:k], 2, axis=1)
            active = _speech_frames(v)
            hop = int(SR * FRAME_S)
            frames = v[:len(active) * hop].reshape(len(active), hop)
            vrms = float(np.sqrt((frames[active] ** 2).mean())) if active.any() else 10 ** (-20 / 20)
            mrms = float(np.sqrt((bed[:k] ** 2).mean())) if k else 0.0
            if mrms > 1e-5:                                  # müzik, konuşmanın MUSIC_BELOW_VOICE_DB altında
                bed *= vrms * 10 ** (-MUSIC_BELOW_VOICE_DB / 20) / mrms
            bed *= _duck_gain(active, n)[:, None]
            fi, fo = min(n, int(0.6 * SR)), min(n, int(1.6 * SR))
            bed[:fi] *= np.linspace(0, 1, fi, dtype=np.float32)[:, None]
            bed[n - fo:] *= np.linspace(1, 0, fo, dtype=np.float32)[:, None]
            mix = mix + bed
        peak = float(np.abs(mix).max()) if n else 0.0
        if peak > 0.98:
            mix *= 0.98 / peak
        raw = self.work / "short_mix_raw.wav"
        _write_wav(raw, mix, SR)
        out = self.work / "short_audio.wav"
        try:
            _run(["ffmpeg", "-y", "-i", str(raw), "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-ar", str(SR),
                  "-ac", "2", "-c:a", "pcm_s16le", str(out)], "short ses yüksekliği", timeout=CUT_TIMEOUT_S)
            got, _ = _read_wav(out)
            if abs(len(got) - n) > 0.1 * SR or _longest_silence(got.mean(1), SR, 0.3, total - END_CARD_S - 0.2)[0] \
                    > _longest_silence(mix.mean(1), SR, 0.3, total - END_CARD_S - 0.2)[0] + 0.3:
                raise RenderError("ses yüksekliği ayarı sesi bozdu")
        except RenderError as e:                             # yedek: Python'da basit kazanç (tepe -1 dBFS)
            log.warning("Short: FFmpeg ses yüksekliği ayarı kullanılamadı (%s), basit kazanç uygulanıyor.", e)
            gain = min(10 ** (-16 / 20) / max(_rms(mix), 1e-6), 0.89 / max(peak, 1e-6))
            _write_wav(out, mix * gain, SR)
        return raw, out

    # ------------------------------------------------------------------ #
    def _compose(self, v_in: Path, audio: Path, ass: Path, total: float) -> None:
        """Görüntü kurgusu; ses hazır dosyadan olduğu gibi eklenir (FFmpeg'de ses filtresi yok)."""
        end_at = total - END_CARD_S
        fc = [
            f"[0:v]tpad=stop_mode=clone:stop_duration={END_CARD_S},split=2[b][f]",
            f"[b]scale=-2:{SH},crop={SW}:{SH},gblur=sigma=40,eq=brightness=-0.12:saturation=0.7[bg]",
            f"[f]scale=-2:{FG_H},crop={SW}:{FG_H}[fg]",
            f"[bg][fg]overlay=0:{FG_Y},drawbox=x=0:y=0:w=iw:h=ih:color=black@0.62:t=fill:"
            f"enable='gte(t,{end_at:.3f})',subtitles=filename={ass.name},format=yuv420p[v]",
        ]
        tmp = self.out.with_suffix(".part.mp4")
        _run(["ffmpeg", "-y", "-i", str(v_in), "-i", str(audio), "-filter_complex", ";".join(fc),
              "-map", "[v]", "-map", "1:a", "-t", f"{total:.3f}", *X264, "-c:a", "aac", "-b:a", "192k",
              "-movflags", "+faststart", str(tmp.resolve())], "short kurgu", cwd=self.work,
             timeout=COMPOSE_TIMEOUT_S)
        tmp.replace(self.out)

    # ------------------------------------------------------------------ #
    def _scan_narration(self) -> list[dict]:
        """Teşhis: belgesel anlatımında cümle içinde uzun sessizlik var mı (seslendirme sorunu)? Yalnızca uyarır."""
        found = []
        try:
            data, sr = _read_wav(self.work / "narration.wav")
            x = data.mean(axis=1)
            for tm in self.r.timings:
                gap, at = _longest_silence(x, sr, tm.speech_start + 0.05, tm.end - 0.05)
                if gap > NARR_MAX_GAP_S:
                    found.append({"sentence": tm.sentence.id, "silence_s": round(gap, 2), "video_at_s": round(at, 1)})
        except Exception as e:                               # teşhis asla işi durdurmaz
            log.warning("Short: anlatım taraması yapılamadı: %s", e)
        for f in found:
            m, s = divmod(int(f["video_at_s"]), 60)
            log.warning("Belgesel anlatımında %.1f sn sessizlik: cümle %s içinde (videoda %d:%02d).",
                        f["silence_s"], f["sentence"], m, s)
        return found

    def check_audio(self, segs: list[Segment], total: float, narr: Path, music: Path | None, mix: Path) -> dict:
        """Yüklemeden önce: anlatım cümle içinde kesilmiş mi, karışımda tam sessizlik var mı, kapanışta müzik var mı.
        Sorun varsa hangi parçada ve hangi saniyede olduğu hata mesajına yazılır."""
        end_at = total - END_CARD_S
        voice, _ = _read_wav(narr)
        vx = voice.mean(axis=1)
        diag: dict = {"narration_gaps": []}
        for s in segs:                                       # konuşma bölgesinin içinde uzun sessizlik = kesik ses
            gap, at = _longest_silence(vx, SR, s.speech_from + 0.05, s.speech_to - 0.05)
            if gap > NARR_MAX_GAP_S:
                diag["narration_gaps"].append({"sentence": s.sid, "silence_s": round(gap, 2), "at_s": round(at, 1)})
        if music is not None:
            mus, _ = _read_wav(music)
            gap, at = _longest_silence(mus.mean(axis=1), SR, 0.3, end_at - 0.2)
            diag["music_longest_silence_s"], diag["music_silence_at_s"] = round(gap, 2), round(at, 1)
        mixed, _ = _read_wav(mix)
        mx = mixed.mean(axis=1)
        longest, at = _longest_silence(mx, SR, 0.3, end_at - 0.2)
        diag["longest_silence_s"], diag["silence_at_s"] = round(longest, 2), round(at, 1)
        card = mx[int((end_at + 0.4) * SR):int((total - 0.6) * SR)]
        diag["end_card_db"] = round(_db(card), 1)
        self.diag = diag
        if diag["narration_gaps"]:
            g = diag["narration_gaps"][0]
            raise RenderError(f"Short ses kontrolü: anlatım cümle {g['sentence']} içinde {g['silence_s']:.1f} sn "
                              f"kesik (Short'un {g['at_s']:.1f}. saniyesi).")
        if longest > MAX_SILENCE_S:
            where = "müzik de sessiz" if diag.get("music_longest_silence_s", 0) > MAX_SILENCE_S else \
                "müzik yok" if music is None else "müzik var ama karışımda kayboldu"
            raise RenderError(f"Short ses kontrolü: {longest:.1f} sn tam sessizlik (Short'un {at:.1f}. saniyesi; "
                              f"{where}).")
        if music is not None and diag["end_card_db"] < SILENCE_DB:
            raise RenderError(f"Short ses kontrolü: kapanış kartında müzik yok ({diag['end_card_db']:.0f} dB).")
        return diag

    def check_final(self, total: float) -> dict:
        """Son dosya (mp4) içindeki ses: karışımdaki gibi mi (kodlama/birleştirmede kayıp yok mu)?"""
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(self.out), "-ac", "1", "-ar", "8000", "-f", "s16le",
                              "-"], stdin=subprocess.DEVNULL, capture_output=True, timeout=120).stdout
        x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
        if len(x) < 8000:
            raise RenderError("Short'ta ses akışı yok.")
        longest, at = _longest_silence(x, 8000, 0.3, total - END_CARD_S - 0.2)
        if longest > MAX_SILENCE_S:
            raise RenderError(f"Short ses kontrolü (mp4): {longest:.1f} sn tam sessizlik (Short'un {at:.1f}. "
                              "saniyesi; karışım sağlamdı, kayıp video birleştirmede).")
        return {"mp4_longest_silence_s": round(longest, 2), "mp4_audio_s": round(len(x) / 8000, 2)}

    def _save_debug(self, files: list[Path | None], error: str) -> None:
        """Hata olursa Short ve ses parçaları kayıtlara (GitHub artifact) konur: sorun dinlenerek görülebilsin."""
        dst = config.OUTPUT_DIR / "short_debug"
        try:
            dst.mkdir(parents=True, exist_ok=True)
            for f in files:
                if f is not None and f.exists():
                    shutil.copy2(f, dst / f.name)
            (dst / "diag.json").write_text(json.dumps({"error": error, **getattr(self, "diag", {})},
                                                      ensure_ascii=False, indent=2), encoding="utf-8")
            log.info("Short teşhis dosyaları kayıtlara eklendi: %s", dst)
        except Exception as e:
            log.warning("Short teşhis dosyaları kaydedilemedi: %s", e)

    # ------------------------------------------------------------------ #
    def render(self) -> Path | None:
        segs = self.segments()
        if not segs:
            log.warning("Short: seçilecek cümle bulunamadı, Short üretilmedi.")
            return None
        narr_gaps = self._scan_narration()
        v_in, a_in = self._cut(segs)
        with wave.open(str(a_in), "rb") as w:
            total = w.getnframes() / w.getframerate() + END_CARD_S
        ass = self._ass(segs, total)
        music = self._music_bed(total)
        raw = mix = None
        try:
            raw, mix = self._mix(a_in, music, total)
            audio = self.check_audio(segs, total, a_in, music, mix)   # sorun varsa RenderError -> Short yüklenmez
            self._compose(v_in, mix, ass, total)
            audio.update(self.check_final(total))
        except RenderError as e:
            self._save_debug([self.out if self.out.exists() else None, a_in, music, raw, mix], str(e))
            raise
        if narr_gaps:
            audio["documentary_narration_gaps"] = narr_gaps
        real = probe_duration(self.out)
        meta = {"duration_s": round(real, 2), "sentences": [s.sid for s in segs], "hook_text": self.short.hook_text,
                "title": self.short.title, "fps": FPS, "size": f"{SW}x{SH}", "audio_check": audio}
        (config.OUTPUT_DIR / "short.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                      encoding="utf-8")
        log.info("Short hazır: %s (%.1f sn, %d cümle)", self.out.name, real, len(segs))
        return self.out


def render_short(plan: ProductionPlan, renderer: VideoRenderer) -> tuple[Path | None, ShortPlan]:
    short = plan.short
    if short is None or not short.sentence_ids:      # eski planlar veya seçim yapılamadıysa
        short = ShortPlan.fallback(plan.acts, plan.seo)
    return ShortRenderer(plan, short, renderer).render(), short
