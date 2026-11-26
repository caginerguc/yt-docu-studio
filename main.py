"""Vor der Realität — uçtan uca otonom belgesel hattı.

  1 Araştırma (kaynak doğrulamalı dossier)
  2 Senaryo + kesit planı + SEO
  3 Medya (anlatım sesi + arşiv/stok/AI görüntü)
  4 Kapak
  5 Render (FFmpeg)
  6 YouTube yükleme (video + kapak + altyazı) ve konu hafızası

Örnekler:
  python main.py --lang de                       # tam çalıştırma, gizlilik UPLOAD_PRIVACY (varsayılan public)
  python main.py --lang de --privacy unlisted
  python main.py --dry-run                       # her şeyi üret, YouTube'a yükleme
  python main.py --plan output/plan.json         # araştırma+senaryoyu atla, var olan planla devam et
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from src import config, history

log = logging.getLogger("makrokontext.main")

_MUSIC_LABEL = {"de": "Musik", "en": "Music", "tr": "Müzik"}


def music_credits(paths: list[Path], lang: str) -> str:
    """assets/music/classical_by/Eser__Sanatçı.mp3 -> atıf satırı (CC BY zorunluluğu)."""
    lines = []
    for p in dict.fromkeys(paths):          # aynı parça birden çok kez çalınabilir
        if p.parent.name != "classical_by":
            continue
        work, _, artist = p.stem.partition("__")
        work = work.replace("_", " ").replace("-", " ").strip()
        artist = artist.replace("_", " ").strip() or "unbekannt"
        lines.append(f"- {work} – {artist} (CC BY)")
    return f"{_MUSIC_LABEL.get(lang, 'Music')}:\n" + "\n".join(lines) if lines else ""


def quality_gate(plan, media_report: dict) -> list[str]:
    """Herkese açık yayını engelleyen nedenler (boşsa video yayına hazır)."""
    reasons = []
    overall = plan.critique.get("overall")
    if isinstance(overall, (int, float)) and overall < config.PUBLIC_MIN_SCORE:
        reasons.append(f"eleştirmen puanı {overall} < {config.PUBLIC_MIN_SCORE}")
    fc = plan.critique.get("factcheck") or {}
    if fc.get("llm_check") not in (None, "ok"):
        reasons.append(f"LLM olgu denetimi çalışmadı ({fc.get('llm_check')})")
    weak = media_report.get("reused_or_blank_share", 0)
    if weak > config.PUBLIC_MAX_WEAK_VISUALS:
        reasons.append(f"tekrar/boş kesit payı %{round(weak * 100)} > %{round(config.PUBLIC_MAX_WEAK_VISUALS * 100)}")
    return reasons


def main() -> int:
    ap = argparse.ArgumentParser(description="Vor der Realität otonom belgesel stüdyosu")
    ap.add_argument("--lang", default=config.DEFAULT_LANG, type=str.lower, choices=sorted(config.LANGS))
    ap.add_argument("--privacy", default=config.UPLOAD_PRIVACY, type=str.lower,
                    choices=["private", "unlisted", "public"])
    ap.add_argument("--dry-run", action="store_true", help="YouTube'a yükleme yapma")
    ap.add_argument("--skip-if-done", action="store_true", help="bugün bu dilde video yüklendiyse çık")
    ap.add_argument("--plan", default=None, help="var olan plan.json ile devam et (1-2. aşamayı atla)")
    args = ap.parse_args()

    config.ensure_dirs()
    config.setup_logging()
    upload = config.UPLOAD_ENABLED and not args.dry_run

    errors, warnings = config.validate(upload=upload)
    for w in warnings:
        log.warning("Uyarı: %s", w)
    if errors:
        for e in errors:
            log.error("Kritik: %s", e)
        return 2
    if args.skip_if_done and history.is_done_today(args.lang):
        log.info("Bugün %s dilinde zaten video üretildi; çıkılıyor.", args.lang)
        return 0
    if upload:   # token/kanal sorunu 30 dk üretimden sonra değil, en başta yakalansın
        from src.youtube_check import preflight
        problem = preflight()
        if problem:
            log.error("Kritik: YouTube yüklemesi yapılamaz, üretim başlatılmadı: %s", problem)
            return 2
        log.info("YouTube ön kontrolü tamam (token ve kanal).")

    t0 = time.monotonic()
    plan = None
    try:
        from src.agents_editorial import build_description
        from src.plan import ProductionPlan

        # 1-2) Araştırma + senaryo
        if args.plan:
            plan = ProductionPlan.load(Path(args.plan))
            log.info("Var olan plan kullanılıyor: %s", plan.working_title)
        else:
            from src.agent_researcher import ResearchAgent
            from src.agents_editorial import EditorialCrew
            from src.llm import LLM
            from src.llm_judge import JudgeLLM

            log.info("=== Vor der Realität başladı (dil: %s, gizlilik: %s) ===", args.lang, args.privacy)
            llm = LLM()
            judge = JudgeLLM(llm)
            llm.secondary = judge        # Gemini kesintisinde (503/kota) tüm metin işleri NVIDIA NIM'e düşer
            dossier = ResearchAgent(llm, args.lang).run()
            plan = EditorialCrew(llm, args.lang, judge=judge).produce(dossier)
            plan.save()
            log.info("Plan: '%s' — %d kelime, ~%.1f dk, %d kesit", plan.seo.title, plan.word_count,
                     plan.est_minutes, plan.shot_count())

        # 3) Medya
        from src.media_generator import MediaGenerator
        media = MediaGenerator(plan)
        media.run()
        media_report = media.report()

        # 4) Kapak
        from src.thumbnail import build_thumbnails
        thumb: Path | None = None
        bg = config.SCENES_DIR / "thumb_bg.jpg"
        if bg.exists():
            thumbs = build_thumbnails(bg, plan.seo.thumbnail_text, plan.seo.thumbnail_tag)
            thumb = thumbs[0] if thumbs else None

        # 5) Render
        from src.video_renderer import VideoRenderer
        renderer = VideoRenderer(plan)
        video = renderer.render()

        # Açıklama: giriş + bölümler + kaynaklar + (CC BY müzik atfı) + yapay zekâ beyanı + hashtag
        starts = [t for t, _ in renderer.chapters]
        description = build_description(plan, starts if renderer.chapters_path.read_text().strip() else None)
        credits = music_credits(renderer.music_paths, plan.language)
        if credits:
            description = description + "\n\n" + credits
        meta = {"title": plan.seo.title, "description": description, "tags": plan.seo.tags,
                "video": str(video), "thumbnail": str(thumb) if thumb else None,
                "subtitles": str(renderer.srt_path), "privacy": args.privacy, "topic_id": plan.topic_id}
        config.METADATA_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        # 6) Yükleme
        if not upload:
            history.set_status(plan.topic_id, "produced")
            log.info("Yükleme kapalı (dry-run). Video: %s", video)
        else:
            from src.youtube_uploader import VideoMeta, YouTubeUploader
            privacy = args.privacy
            if privacy == "public":
                reasons = quality_gate(plan, media_report)
                if reasons and not config.QUALITY_GATE:
                    log.warning("Kalite uyarısı (kapı kapalı, yine de public yükleniyor): %s", "; ".join(reasons))
                elif reasons:
                    privacy = "private"
                    log.warning("Kalite kapısı: video PRIVATE yükleniyor, elle inceleyip yayına alın. Nedenler: %s",
                                "; ".join(reasons))
            up = YouTubeUploader()
            vid = up.upload_video(video, VideoMeta(plan.seo.title, description, plan.seo.tags,
                                                   plan.language, privacy))
            up.set_thumbnail(vid, thumb)
            up.add_captions(vid, renderer.srt_path, plan.language)
            actual_privacy = getattr(up, "last_privacy", privacy)
            history.set_status(plan.topic_id, "uploaded", video_id=vid, url=f"https://youtu.be/{vid}",
                               privacy=actual_privacy, requested_privacy=privacy)
            meta["video_id"] = vid
            meta["privacy"] = actual_privacy
            config.METADATA_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        log.info("=== Tamamlandı: %.1f dk ===", (time.monotonic() - t0) / 60)
        return 0

    except Exception as e:
        log.exception("Üretim başarısız: %s", e)
        if plan is not None:
            history.set_status(plan.topic_id, "failed", error=str(e)[:300])
        return 1


if __name__ == "__main__":
    sys.exit(main())
