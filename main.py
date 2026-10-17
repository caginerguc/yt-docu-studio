import argparse
import logging
import shutil
from src.config import OUTPUT_DIR, TARGET_LANGUAGE
from src.agent_researcher import NewsIntelligenceAgent
from src.agents_editorial import EditorialProductionCrew
from src.media_generator import MediaGenerator
from src.video_renderer import VideoRenderer
from src.youtube_uploader import YouTubeUploader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Main_Orchestrator")


def clean_temp_directories():
    """Yeni prodüksiyon öncesi önceki videonun sahne, ses ve klip kalıntılarını temizler."""
    for folder in ["scenes", "audio", "clips"]:
        dir_path = OUTPUT_DIR / folder
        if dir_path.exists():
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)


def run_autonomous_studio(language: str, privacy: str, skip_upload: bool = False):
    logger.info("==========================================================")
    logger.info(f"🚀 OTONOM YOUTUBE BELGESEL STÜDYOSU BAŞLATILDI (DİL: {language})")
    logger.info("==========================================================")

    clean_temp_directories()

    # 1. AŞAMA: Haber Tarama & Seçim (Ajan 1)
    agent1 = NewsIntelligenceAgent(language=language)
    selected_topic = agent1.select_best_topic(auto_save_to_memory=True)

    # 2. AŞAMA: Senaryo, Görsel Prompt, Müzik & SEO (Ajan 2, 3 ve 4)
    crew = EditorialProductionCrew(language=language)
    plan = crew.create_production_plan(selected_topic, save_to_disk=True)

    # 3. AŞAMA: 16:9 Flux Görsel & Edge-TTS Ses/Altyazı Üretimi
    media_gen = MediaGenerator(plan)
    media_gen.run_pipeline()

    # 4. AŞAMA: FFmpeg 1080p Ken Burns, -24dB Miksaj & Altyazı Gömme
    renderer = VideoRenderer(plan)
    final_video = renderer.render_full_video()
    thumbnail_img = OUTPUT_DIR / "thumbnail.jpg"

    # 5. AŞAMA: YouTube Data API v3 Yükleme
    if skip_upload:
        logger.info("Skip-upload aktif: YouTube yüklemesi atlandı.")
        return

    uploader = YouTubeUploader()
    if uploader.is_configured():
        video_id = uploader.upload_documentary(
            video_path=final_video,
            thumbnail_path=thumbnail_img,
            plan=plan,
            privacy_status=privacy
        )
        logger.info(f"🎉 SÜREÇ TAMAMLANDI! Yayındaki Video: https://youtu.be/{video_id}")
    else:
        logger.warning("YouTube API kimlik bilgileri tanımlı değil, video yalnızca yerel diske kaydedildi.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Otonom YouTube Belgesel Stüdyosu Orkestratörü")
    parser.add_argument("--lang", type=str, default=TARGET_LANGUAGE, help="Hedef dil: DE, EN veya TR")
    parser.add_argument("--privacy", type=str, default="private", choices=["private", "unlisted", "public"], help="YouTube gizlilik ayarı")
    parser.add_argument("--skip-upload", action="store_true", help="YouTube'a yüklemeden sadece videoyu üretir")
    args = parser.parse_args()

    run_autonomous_studio(language=args.lang.upper(), privacy=args.privacy, skip_upload=args.skip_upload)
