"""Aşama 5 — Sinematik kurgu (FFmpeg).

Girdi : ProductionPlan + output/audio/<sentence_id>.mp3 + output/scenes/<sentence_id>_shot<N>.(mp4|jpg)
Çıktı : output/final_documentary.mp4, output/subtitles.srt, output/chapters.txt

Akış
  1. Her cümle için anlatım segmenti = [duraklama sessizliği + cümle sesi]  (WAV, 48 kHz)
     -> gerçek süreler ölçülür; görüntü bu sürelere kare hassasiyetinde bağlanır (kayma yok).
  2. Her kesit ayrı klip olarak üretilir (paralel, önbellekli):
       * stok video  : 1080p'ye ölçek/kırp, gerekirse döngü
       * durağan     : dikey/kare arşiv fotoğrafı bulanık arka plan üzerine oturtulur (kafa kesilmez),
                       4K supersample üzerinden titremesiz Ken Burns (zoom / pan)
       * ortak görünüm: hafif kontrast, düşük doygunluk, film greni, vinyet
       * perde başları: siyahtan açılış; perde sonları: siyaha kararma
  3. Klipler yeniden kodlamadan birleştirilir (concat demuxer).
  4. Ses: anlatım + klasik müzik (kısık, konuşmada otomatik "ducking", fade in/out) -> -14 LUFS.
  5. Altyazı (SRT) ve YouTube bölüm zaman damgaları üretilir.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFilter

from src import config
from src.plan import PlanSentence, ProductionPlan

log = logging.getLogger("makrokontext.renderer")

FPS = config.FPS
W, H = config.VIDEO_W, config.VIDEO_H
SR = 48000
INTRO_S = 0.8                    # ilk cümleden önce
OUTRO_S = 4.0                    # son cümleden sonra (müzik + kararma)
ACT_BREAK_S = 0.9                # yeni perde başında ek nefes
SHORT_PAUSE_S = 0.35             # pause_before == "short" için ek
FADE_S = config.CROSSFADE_S      # perde geçişlerinde siyaha kararma/açılma
KB_ZOOM = 0.12                   # Ken Burns yakınlaşma miktarı
MUSIC_EXTS = (".mp3", ".mpga", ".ogg", ".oga", ".flac", ".wav", ".m4a", ".opus")
LOOK = "eq=contrast=1.06:brightness=-0.02:saturation=0.86,noise=alls=5:allf=t,vignette=PI/5"
X264 = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-r", str(FPS), "-g", str(FPS * 2), "-video_track_timescale", str(FPS * 1000)]
SUB_MAX_WORDS = 12
# Müzik yatağı kaydın kendi seviyesinden bağımsız olarak bu yüksekliğe getirilir; anlatım ~-14 LUFS.
# -30 = konuşmanın ~14 dB altı (aralarda duyulur, konuşmada ayrıca kısılır). Daha kısık için -32.
MUSIC_LUFS = float(os.environ.get("MUSIC_LUFS", "-30"))


class RenderError(RuntimeError):
    pass


@dataclass
class ShotJob:
    key: str
    src: Path
    is_video: bool
    frames: int
    camera: str
    fade_in: float
    fade_out: float
    out: Path


@dataclass
class SentenceTiming:
    sentence: PlanSentence
    start: float          # segment başlangıcı (duraklama dahil), saniye
    speech_start: float   # konuşmanın başladığı an
    end: float


def _run(cmd: list[str], what: str) -> None:
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        tail = "\n".join(res.stderr.strip().splitlines()[-12:])
        raise RenderError(f"FFmpeg hatası ({what}), kod {res.returncode}:\n{tail}")


def probe_duration(path: Path) -> float:
    res = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(res.stdout.strip())
    except ValueError as e:
        raise RenderError(f"Süre okunamadı: {path.name} ({res.stderr.strip()[:200]})") from e


def _fmt_srt(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_chapter(t: float) -> str:
    t = int(t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class VideoRenderer:
    def __init__(self, plan: ProductionPlan):
        self.plan = plan
        self.clips_dir = config.CLIPS_DIR
        self.work_dir = config.CACHE_DIR / "render"
        for d in (self.clips_dir, self.work_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.final_video = config.OUTPUT_DIR / "final_documentary.mp4"
        self.srt_path = config.OUTPUT_DIR / "subtitles.srt"
        self.chapters_path = config.OUTPUT_DIR / "chapters.txt"
        self.chapters: list[tuple[float, str]] = []
        self.timings: list[SentenceTiming] = []
        self.duration = 0.0
        self.music_used: list[str] = []
        self.music_paths: list[Path] = []
        self._rng = random.Random(int(hashlib.sha1(plan.topic_id.encode()).hexdigest()[:8], 16))

    # ================================================================== #
    # 1) Anlatım segmentleri
    # ================================================================== #
    def _gap_before(self, s: PlanSentence, first_in_act: bool, first_overall: bool) -> float:
        if first_overall:
            return INTRO_S
        gap = config.SENTENCE_GAP_S
        if s.pause_before == "short":
            gap += SHORT_PAUSE_S
        elif s.pause_before == "long":
            gap += config.TWIST_GAP_S
        if first_in_act:
            gap += ACT_BREAK_S
        return gap

    def build_narration(self) -> Path:
        segs: list[Path] = []
        t = 0.0
        first_overall = True
        for act in self.plan.acts:
            for i, s in enumerate(act.sentences):
                src = config.AUDIO_DIR / f"{s.id}.mp3"
                if not src.exists():
                    raise RenderError(f"Anlatım sesi eksik: {src.name} (media_generator çalıştı mı?)")
                gap = self._gap_before(s, i == 0, first_overall)
                seg = self.work_dir / f"seg_{s.id}.wav"
                _run(["ffmpeg", "-y", "-f", "lavfi", "-t", f"{gap:.3f}", "-i", f"anullsrc=r={SR}:cl=mono",
                      "-i", str(src), "-filter_complex",
                      f"[1:a]aresample={SR},aformat=channel_layouts=mono,"
                      "silenceremove=start_periods=1:start_threshold=-50dB,areverse,"
                      "silenceremove=start_periods=1:start_threshold=-50dB,areverse[v];"
                      "[0:a][v]concat=n=2:v=0:a=1[a]",
                      "-map", "[a]", "-c:a", "pcm_s16le", str(seg)], f"segment {s.id}")
                d = probe_duration(seg)
                if i == 0:
                    self.chapters.append((t, act.chapter_title or act.name))
                self.timings.append(SentenceTiming(s, t, t + gap, t + d))
                t += d
                segs.append(seg)
                first_overall = False
        tail = self.work_dir / "tail.wav"
        _run(["ffmpeg", "-y", "-f", "lavfi", "-t", f"{OUTRO_S:.3f}", "-i", f"anullsrc=r={SR}:cl=mono",
              "-c:a", "pcm_s16le", str(tail)], "tail")
        segs.append(tail)
        self.duration = t + OUTRO_S

        lst = self.work_dir / "narration.txt"
        lst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in segs), encoding="utf-8")
        out = self.work_dir / "narration.wav"
        _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c:a", "pcm_s16le", str(out)],
             "narration concat")
        log.info("Anlatım hazır: %d cümle, %.1f dk", len(self.timings), self.duration / 60)
        return out

    # ================================================================== #
    # 2) Kesit klipleri
    # ================================================================== #
    def _shot_source(self, sid: str, idx: int) -> tuple[Path, bool]:
        vid = config.SCENES_DIR / f"{sid}_shot{idx}.mp4"
        img = config.SCENES_DIR / f"{sid}_shot{idx}.jpg"
        if vid.exists() and vid.stat().st_size > 100_000:
            return vid, True
        if img.exists() and img.stat().st_size > config.MIN_IMAGE_BYTES:
            return img, False
        fallback = self.work_dir / "black.jpg"
        if not fallback.exists():
            Image.new("RGB", (W, H), (12, 14, 18)).save(fallback, "JPEG", quality=90)
        log.warning("Kesit görseli eksik, siyah kare: %s_shot%d", sid, idx)
        return fallback, False

    def _plan_jobs(self) -> list[ShotJob]:
        jobs: list[ShotJob] = []
        n_t = len(self.timings)
        act_last = {}
        for k, tm in enumerate(self.timings):
            act_last[tm.sentence.act_id] = k
        for k, tm in enumerate(self.timings):
            s = tm.sentence
            end = self.duration if k == n_t - 1 else tm.end       # son cümle outro'yu kapsar
            f0, f1 = round(tm.start * FPS), round(end * FPS)
            shots = s.shots or []
            n = max(1, len(shots))
            bounds = [f0 + round((f1 - f0) * j / n) for j in range(n + 1)]
            first_of_act = k == 0 or self.timings[k - 1].sentence.act_id != s.act_id
            last_of_act = act_last[s.act_id] == k
            for j in range(n):
                cam = shots[j].camera if j < len(shots) else "zoom_in"
                src, is_vid = self._shot_source(s.id, j + 1)
                frames = max(1, bounds[j + 1] - bounds[j])
                fi = (1.2 if k == 0 else FADE_S) if (first_of_act and j == 0) else 0.0
                fo = (2.5 if k == n_t - 1 else FADE_S) if (last_of_act and j == n - 1) else 0.0
                h = hashlib.sha1(f"{src.name}|{src.stat().st_size}|{frames}|{cam}|{fi}|{fo}|v2".encode()).hexdigest()[:10]
                jobs.append(ShotJob(f"{s.id}_{j + 1}", src, is_vid, frames, cam, fi, fo,
                                    self.clips_dir / f"{s.id}_shot{j + 1}_{h}.mp4"))
        return jobs

    def _prepare_still(self, src: Path) -> Path:
        """Her durağanı 3840x2160'a getirir; en-boy oranı farklıysa bulanık arka plan + ortalanmış ön plan."""
        out = self.work_dir / f"still_{src.stem}_{src.stat().st_size}.jpg"
        if out.exists():
            return out
        tw, th = config.KB_SUPERSAMPLE_W, config.KB_SUPERSAMPLE_H
        with Image.open(src) as im:
            im = im.convert("RGB")
            ratio = im.width / im.height
            if abs(ratio - tw / th) / (tw / th) < 0.18:            # neredeyse 16:9 -> kırparak doldur
                scale = max(tw / im.width, th / im.height)
                im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
                l, t = (im.width - tw) // 2, (im.height - th) // 2
                canvas = im.crop((l, t, l + tw, t + th))
            else:                                                   # portre/kare -> kafa kesilmesin
                bg_scale = max(tw / im.width, th / im.height)
                bg = im.resize((round(im.width * bg_scale), round(im.height * bg_scale)), Image.BILINEAR)
                l, t = (bg.width - tw) // 2, (bg.height - th) // 2
                bg = bg.crop((l, t, l + tw, t + th)).filter(ImageFilter.GaussianBlur(60))
                bg = Image.eval(bg, lambda v: int(v * 0.45))
                fg_scale = min(tw * 0.92 / im.width, th * 0.92 / im.height)
                fg = im.resize((round(im.width * fg_scale), round(im.height * fg_scale)), Image.LANCZOS)
                bg.paste(fg, ((tw - fg.width) // 2, (th - fg.height) // 2))
                canvas = bg
            canvas.save(out, "JPEG", quality=93)
        return out

    def _kenburns(self, cam: str, frames: int) -> str:
        n = max(frames - 1, 1)
        p = f"(on/{n})"
        if cam == "zoom_out":
            z, x, y = f"{1 + KB_ZOOM}-{KB_ZOOM}*{p}", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
        elif cam == "pan_left_to_right":
            z, x, y = f"{1 + KB_ZOOM * 0.8}", f"(iw-iw/zoom)*{p}", "ih/2-(ih/zoom/2)"
        elif cam == "pan_right_to_left":
            z, x, y = f"{1 + KB_ZOOM * 0.8}", f"(iw-iw/zoom)*(1-{p})", "ih/2-(ih/zoom/2)"
        else:  # zoom_in
            z, x, y = f"1+{KB_ZOOM}*{p}", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
        return f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={W}x{H}:fps={FPS}"

    def _fades(self, job: ShotJob) -> str:
        dur = job.frames / FPS
        parts = []
        if job.fade_in > 0:
            parts.append(f"fade=t=in:st=0:d={min(job.fade_in, dur / 2):.3f}")
        if job.fade_out > 0:
            d = min(job.fade_out, dur / 2)
            parts.append(f"fade=t=out:st={dur - d:.3f}:d={d:.3f}")
        return ("," + ",".join(parts)) if parts else ""

    def _render_job(self, job: ShotJob) -> Path:
        if job.out.exists() and job.out.stat().st_size > 1000:
            return job.out
        tmp = job.out.with_suffix(".part.mp4")
        dur = job.frames / FPS
        if job.is_video:
            src_d = probe_duration(job.src)
            start = 0.0
            if src_d > dur + 1.0:      # uzun klipte her seferinde baştan başlama: rastgele ama tekrarlanabilir kesit
                start = random.Random(job.key).uniform(0, min(src_d - dur - 0.5, src_d * 0.5))
            vf = (f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,fps={FPS},"
                  f"{LOOK}{self._fades(job)},format=yuv420p")
            cmd = ["ffmpeg", "-y", "-ss", f"{start:.2f}", "-stream_loop", "-1", "-i", str(job.src),
                   "-vf", vf, "-frames:v", str(job.frames), "-an", *X264, "-threads", "2", str(tmp)]
        else:
            still = self._prepare_still(job.src)
            vf = f"{self._kenburns(job.camera, job.frames)},setsar=1,{LOOK}{self._fades(job)},format=yuv420p"
            cmd = ["ffmpeg", "-y", "-i", str(still), "-vf", vf, "-frames:v", str(job.frames),
                   "-an", *X264, "-threads", "2", str(tmp)]
        _run(cmd, f"klip {job.key} ({dur:.1f} sn)")
        tmp.replace(job.out)
        return job.out

    def render_clips(self, jobs: list[ShotJob]) -> list[Path]:
        workers = max(1, min(4, (os.cpu_count() or 2) // 2 or 1))
        log.info("%d kesit klibi işleniyor (%d paralel iş)...", len(jobs), workers)
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            outs = list(ex.map(self._render_job, jobs))
        log.info("Klipler hazır (%.1f dk).", (time.monotonic() - t0) / 60)
        return outs

    def concat_video(self, clips: list[Path]) -> Path:
        lst = self.work_dir / "clips.txt"
        lst.write_text("".join(f"file '{c.resolve().as_posix()}'\n" for c in clips), encoding="utf-8")
        out = self.work_dir / "video_only.mp4"
        _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)],
             "video concat")
        return out

    # ================================================================== #
    # 3) Müzik
    # ================================================================== #
    def pick_music(self, need_s: float) -> list[Path]:
        """Havuz sırası: assets/music/<mood>/  ->  classical/ + classical_by/  ->  assets/music/ kökü."""
        mood = getattr(self.plan.seo, "music_mood", "") or ""

        def files(*dirs: Path) -> list[Path]:
            return sorted(p for d in dirs if d.is_dir() for p in d.iterdir()
                          if p.is_file() and p.suffix.lower() in MUSIC_EXTS)

        pool: list[Path] = []
        for group in ((config.MUSIC_DIR / mood,) if mood else (),
                      (config.MUSIC_DIR / "classical", config.MUSIC_DIR / "classical_by"),
                      (config.MUSIC_DIR,)):
            pool = files(*group)
            if pool:
                break
        if not pool:
            return []
        self._rng.shuffle(pool)
        chosen, total = [], 0.0
        while total < need_s and len(chosen) < 12:
            p = pool[len(chosen) % len(pool)]
            d = probe_duration(p)
            if d < 20:
                if len(pool) == 1:
                    break
                pool.remove(p)
                continue
            chosen.append(p)
            total += d - 4.0
        return chosen

    def mix_audio(self, narration: Path) -> Path:
        out = self.work_dir / "mix.m4a"
        tracks = self.pick_music(self.duration)
        self.music_used = [p.name for p in tracks]
        self.music_paths = tracks
        T = self.duration
        loud = "loudnorm=I=-14:TP=-1.5:LRA=11"
        if not tracks:
            log.warning("assets/music içinde müzik yok: yalnızca anlatım kullanılacak.")
            _run(["ffmpeg", "-y", "-i", str(narration), "-af", f"aresample={SR},{loud}",
                  "-ac", "2", "-c:a", "aac", "-b:a", "192k", str(out)], "audio (müziksiz)")
            return out
        log.info("Müzik: %s", ", ".join(self.music_used))
        inputs: list[str] = ["-i", str(narration)]
        for p in tracks:
            inputs += ["-i", str(p)]
        norm = f"aresample={SR},aformat=sample_fmts=fltp:channel_layouts=stereo"
        fc = [f"[0:a]{norm},asplit=2[voice][key]"]
        labels = []
        for i in range(1, len(tracks) + 1):
            fc.append(f"[{i}:a]{norm}[m{i}]")
            labels.append(f"[m{i}]")
        cur = labels[0]
        for i, lab in enumerate(labels[1:], 2):           # parçalar arasında 4 sn yumuşak geçiş
            fc.append(f"{cur}{lab}acrossfade=d=4:c1=tri:c2=tri[x{i}]")
            cur = f"[x{i}]"
        fc.append(f"{cur}aloop=loop=-1:size=2147483647,atrim=0:{T:.3f},"
                  f"loudnorm=I={MUSIC_LUFS}:TP=-8:LRA=14,aresample={SR},"
                  f"afade=t=in:d=3,afade=t=out:st={max(0.0, T - 5):.3f}:d=5[bed]")
        # Konuşma varken müzik otomatik kısılır (ducking); cümle aralarında yavaşça geri gelir.
        fc.append("[bed][key]sidechaincompress=threshold=0.015:ratio=6:attack=40:release=900:makeup=1[duck]")
        fc.append(f"[voice][duck]amix=inputs=2:duration=first:normalize=0,{loud}[out]")
        _run(["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(fc), "-map", "[out]",
              "-t", f"{T:.3f}", "-c:a", "aac", "-b:a", "192k", str(out)], "audio mix")
        return out

    # ================================================================== #
    # 4) Altyazı ve bölümler
    # ================================================================== #
    def write_srt(self) -> Path:
        blocks, n = [], 0
        for tm in self.timings:
            words = tm.sentence.text.split()
            chunks = [words[i:i + SUB_MAX_WORDS] for i in range(0, len(words), SUB_MAX_WORDS)]
            if len(chunks) > 1 and len(chunks[-1]) < 4:          # çok kısa son parça öncekiyle birleşsin
                chunks[-2] += chunks.pop()
            total_chars = sum(len(" ".join(c)) for c in chunks) or 1
            t = tm.speech_start
            span = max(0.5, tm.end - tm.speech_start)
            for c in chunks:
                txt = " ".join(c)
                d = span * len(txt) / total_chars
                n += 1
                if len(txt) > 42:                                  # iki satıra böl
                    mid = len(c) // 2
                    txt = " ".join(c[:mid]) + "\n" + " ".join(c[mid:])
                blocks.append(f"{n}\n{_fmt_srt(t)} --> {_fmt_srt(t + d - 0.05)}\n{txt}\n")
                t += d
        self.srt_path.write_text("\n".join(blocks), encoding="utf-8")
        return self.srt_path

    def write_chapters(self) -> Path:
        lines = [f"{_fmt_chapter(t if i else 0)} {title}" for i, (t, title) in enumerate(self.chapters)]
        valid = len(self.chapters) >= 3 and all(b[0] - a[0] >= 10 for a, b in zip(self.chapters, self.chapters[1:]))
        self.chapters_path.write_text("\n".join(lines) if valid else "", encoding="utf-8")
        return self.chapters_path

    # ================================================================== #
    def render(self) -> Path:
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise RenderError("ffmpeg/ffprobe bulunamadı.")
        t0 = time.monotonic()
        log.info("Render başlıyor: %s", self.plan.working_title)
        narration = self.build_narration()
        jobs = self._plan_jobs()
        clips = self.render_clips(jobs)
        video = self.concat_video(clips)
        audio = self.mix_audio(narration)
        tmp = self.final_video.with_suffix(".part.mp4")
        _run(["ffmpeg", "-y", "-i", str(video), "-i", str(audio), "-map", "0:v", "-map", "1:a",
              "-c:v", "copy", "-c:a", "copy", "-t", f"{self.duration:.3f}", "-movflags", "+faststart", str(tmp)],
             "final mux")
        tmp.replace(self.final_video)
        self.write_srt()
        self.write_chapters()
        real = probe_duration(self.final_video)
        stills = sum(1 for j in jobs if not j.is_video)
        meta = {"duration_s": round(real, 2), "shots": len(jobs), "video_shots": len(jobs) - stills,
                "still_shots": stills, "music": self.music_used,
                "render_minutes": round((time.monotonic() - t0) / 60, 1)}
        (config.OUTPUT_DIR / "render.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                       encoding="utf-8")
        log.info("Render tamam: %s (%.1f dk video, %.1f dk işlem)", self.final_video.name, real / 60,
                 meta["render_minutes"])
        return self.final_video
