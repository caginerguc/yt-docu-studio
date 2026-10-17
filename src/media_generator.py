import asyncio
import json
import logging
import time
from pathlib import Path
from urllib.parse import quote
import requests
import edge_tts

from src.config import OUTPUT_DIR, get_active_config
from src.agents_editorial import ProductionPlanSchema

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Stage3_MediaGenerator")


class MediaGenerator:
    def __init__(self, plan: ProductionPlanSchema):
        self.plan = plan
        self.config = get_active_config(plan.language)
        self.voice = self.config["voice"]
        self.scenes_dir = OUTPUT_DIR / "scenes"
        self.audio_dir = OUTPUT_DIR / "audio"
        self.scenes_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def download_pollinations_image(self, prompt: str, output_path: Path, seed: int = 42, retries: int = 5) -> Path:
        """Pollinations.ai üzerinden 16:9 görsel indirir (402/429 Rate-Limit ve model fallback korumalı)."""
        if output_path.exists() and output_path.stat().st_size > 10000:
            logger.info(f"Görsel zaten mevcut, atlanıyor: {output_path.name}")
            return output_path

        enhanced_prompt = f"{prompt}, cinematic 16:9 widescreen, no text, no watermark, photorealistic, 8k"
        encoded_prompt = quote(enhanced_prompt)

        # İlk denemelerde flux, eğer 402/429 verirse turbo/varsayılan modele geçiş (fallback)
        models_to_try = ["flux", "turbo", "flux", "turbo", "flux"]

        for attempt in range(1, retries + 1):
            model_name = models_to_try[attempt - 1]
            # 1280x720 (16:9) ücretsiz katman limitlerine takılmaz, FFmpeg 1080p'ye upscale eder
            url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1280&height=720&seed={seed + attempt}&model={model_name}&nologo=true"
            try:
                logger.info(f"Görsel indiriliyor ({output_path.name} | Model: {model_name}) - Deneme {attempt}/{retries}...")
                response = requests.get(
                    url,
                    timeout=90,
                    headers={"User-Agent": f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/{537 + attempt}.36"}
                )
                
                if response.status_code in (402, 429):
                    wait_sec = 15 * attempt
                    logger.warning(f"Pollinations Hız Sınırı (HTTP {response.status_code}). {wait_sec} sn beklenip tekrar denenecek...")
                    time.sleep(wait_sec)
                    continue

                response.raise_for_status()
                
                with open(output_path, "wb") as f:
                    f.write(response.content)
                
                size_kb = output_path.stat().st_size / 1024
                logger.info(f"Görsel kaydedildi: {output_path.name} ({size_kb:.1f} KB)")
                return output_path
            except Exception as e:
                logger.warning(f"Görsel indirme uyarısı ({output_path.name}): {e}")
                if attempt < retries:
                    time.sleep(10 * attempt)
                else:
                    raise RuntimeError(f"Görsel {retries} denemeye rağmen indirilemedi: {output_path.name}")

    async def _generate_single_tts_and_srt(self, text: str, mp3_path: Path, srt_path: Path):
        """Edge-TTS ile tek bir sahnenin sesini (.mp3) ve senkronize altyazısını (.srt) üretir."""
        if mp3_path.exists() and srt_path.exists() and mp3_path.stat().st_size > 1000:
            logger.info(f"Ses ve Altyazı zaten mevcut, atlanıyor: {mp3_path.name}")
            return

        communicate = edge_tts.Communicate(text=text, voice=self.voice, rate="-2%")
        submaker = edge_tts.SubMaker()

        with open(mp3_path, "wb") as audio_file:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_file.write(chunk["data"])
                elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                    submaker.feed(chunk)

        srt_content = submaker.get_srt()
        with open(srt_path, "w", encoding="utf-8") as srt_file:
            srt_file.write(srt_content)

        logger.info(f"Ses ve Altyazı üretildi: {mp3_path.name} & {srt_path.name}")

    async def generate_all_audio(self):
        """Tüm sahnelerin ses ve altyazılarını sırayla üretir."""
        for scene in self.plan.scenes:
            idx = scene.scene_number
            mp3_path = self.audio_dir / f"scene_{idx}.mp3"
            srt_path = self.audio_dir / f"scene_{idx}.srt"
            await self._generate_single_tts_and_srt(scene.narration_text, mp3_path, srt_path)

    def generate_all_visuals(self):
        """Tüm sahnelerin 16:9 görsellerini ve YouTube Thumbnail kapak görselini üretir."""
        base_seed = int(time.time()) % 10000
        for scene in self.plan.scenes:
            idx = scene.scene_number
            img_path = self.scenes_dir / f"scene_{idx}.jpg"
            need_sleep = not img_path.exists()
            self.download_pollinations_image(
                prompt=scene.visual_prompt_en,
                output_path=img_path,
                seed=base_seed + idx
            )
            if need_sleep:
                logger.info("Rate-limit koruması için 10 saniye bekleniyor...")
                time.sleep(10)

        thumb_path = OUTPUT_DIR / "thumbnail.jpg"
        self.download_pollinations_image(
            prompt=self.plan.seo.thumbnail_prompt_en,
            output_path=thumb_path,
            seed=base_seed + 99
        )

    def run_pipeline(self):
        """Görsel, Ses ve Altyazı üretimini uçtan uca çalıştırır."""
        logger.info(f"=== Medya Üretimi Başlıyor (Ses: {self.voice}) ===")
        asyncio.run(self.generate_all_audio())
        self.generate_all_visuals()
        logger.info("=== Tüm Görsel, Ses ve Altyazı Dosyaları Başarıyla Üretildi! ===")


if __name__ == "__main__":
    plan_file = OUTPUT_DIR / "latest_plan.json"
    if not plan_file.exists():
        raise FileNotFoundError("latest_plan.json bulunamadı! Önce Ajan 2-3-4'ü çalıştırın.")

    with open(plan_file, "r", encoding="utf-8") as f:
        plan_data = json.load(f)

    plan_obj = ProductionPlanSchema.model_validate(plan_data)
    generator = MediaGenerator(plan_obj)
    generator.run_pipeline()
