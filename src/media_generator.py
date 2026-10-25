import asyncio
import io
import json
import logging
import os
import time
from pathlib import Path
from urllib.parse import quote
import requests
import edge_tts
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from google import genai
from google.genai import types

from src.config import OUTPUT_DIR, get_active_config, GEMINI_API_KEY, POLLINATIONS_API_KEY
from src.agents_editorial import ProductionPlanSchema

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Stage3_MediaGenerator")


class MediaGenerator:
    def __init__(self, plan: ProductionPlanSchema):
        self.plan = plan
        self.config = get_active_config(plan.language)
        self.voice = self.config["voice"]
        self.fallback_voices = self.config.get("fallback_voices", [self.voice])
        self.pollinations_token = POLLINATIONS_API_KEY
        self.scenes_dir = OUTPUT_DIR / "scenes"
        self.audio_dir = OUTPUT_DIR / "audio"
        self.scenes_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        
        if GEMINI_API_KEY:
            self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        else:
            self.gemini_client = None

    def download_image_pollinations(self, prompt: str, output_path: Path, seed: int = 42) -> Path:
        if output_path.exists() and output_path.stat().st_size > 10000:
            logger.info(f"Görsel zaten mevcut, atlanıyor: {output_path.name}")
            return output_path

        enhanced_prompt = f"{prompt}, cinematic 16:9 widescreen, photorealistic, 8k, no text"
        encoded_prompt = quote(enhanced_prompt)
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        if self.pollinations_token:
            headers["Authorization"] = f"Bearer {self.pollinations_token}"

        for model_name in ["flux", "turbo"]:
            for attempt in range(1, 3):
                current_seed = seed + (attempt * 7)
                url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1920&height=1080&seed={current_seed}&model={model_name}&nologo=true"
                if self.pollinations_token:
                    url += f"&key={self.pollinations_token}"

                try:
                    logger.info(f"Pollinations ({model_name}, deneme {attempt}) indiriliyor: {output_path.name}...")
                    response = requests.get(url, timeout=75, headers=headers)
                    if response.status_code == 200 and len(response.content) > 8000:
                        with open(output_path, "wb") as f:
                            f.write(response.content)
                        return output_path
                    logger.warning(f"Pollinations HTTP {response.status_code} döndü, bekleniyor...")
                except Exception as e:
                    logger.warning(f"Pollinations bağlantı uyarısı ({model_name}): {e}")
                time.sleep(4 * attempt)

        raise RuntimeError("Pollinations tüm denemelerde başarısız oldu.")

    def create_studio_fallback_canvas(self, output_path: Path, scene_title: str = "MAKROKONTEXT"):
        """Tüm dış görsel API'leri çökerse sistemi durdurmamak için kurumsal sinematik arka plan üretir."""
        logger.info(f"Kurumsal yedek stüdyo tuvali oluşturuluyor: {output_path.name}")
        img = Image.new("RGB", (1920, 1080), (12, 18, 28))
        draw = ImageDraw.Draw(img)
        for y in range(1080):
            r = int(10 + (y / 1080) * 18)
            g = int(16 + (y / 1080) * 28)
            b = int(28 + (y / 1080) * 45)
            draw.line([(0, y), (1920, y)], fill=(r, g, b))
        for x in range(0, 1920, 120):
            draw.line([(x, 0), (x, 1080)], fill=(25, 38, 58), width=1)
        for y in range(0, 1080, 120):
            draw.line([(0, y), (1920, y)], fill=(25, 38, 58), width=1)
        img.save(output_path, "JPEG", quality=92)
        return output_path

    def download_safe(self, prompt: str, output_path: Path, seed: int = 42):
        try:
            self.download_image_pollinations(prompt, output_path, seed)
        except Exception as e:
            logger.warning(f"Görsel API uyarısı ({e}). Sistemin durmaması için stüdyo tuvali devreye giriyor...")
            self.create_studio_fallback_canvas(output_path)

    def generate_professional_thumbnail(self, raw_thumb_path: Path, title: str):
        """Pillow ile ham arka plan üzerine yüksek kontrastlı YouTube tipografisi ve filigran basar."""
        try:
            logger.info("Pillow ile profesyonel YouTube Thumbnail tipografisi oluşturuluyor...")
            img = Image.open(raw_thumb_path).convert("RGBA")
            img = img.resize((1280, 720), Image.Resampling.LANCZOS)
            
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.25)
            
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw_overlay = ImageDraw.Draw(overlay)
            for y in range(360, 720):
                alpha = int(185 * ((y - 360) / 360))
                draw_overlay.line([(0, y), (1280, y)], fill=(0, 0, 0, alpha))
            
            img = Image.alpha_composite(img, overlay)
            draw = ImageDraw.Draw(img)

            font_title = None
            font_brand = None
            for font_name in [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "impact.ttf",
                "arialbd.ttf",
                "DejaVuSans-Bold.ttf"
            ]:
                try:
                    font_title = ImageFont.truetype(font_name, 68)
                    font_brand = ImageFont.truetype(font_name, 32)
                    break
                except Exception:
                    continue
            
            if not font_title:
                font_title = ImageFont.load_default()
                font_brand = ImageFont.load_default()

            clean_title = title.split(":")[0].replace("?", "").replace("!", "").strip().upper()
            words = clean_title.split()
            display_text = " ".join(words[:4]) if len(words) > 4 else clean_title

            draw.text((50, 480), display_text, font=font_title, fill="#FFE600", stroke_width=6, stroke_fill="#000000")
            draw.rectangle([(50, 40), (340, 92)], fill=(225, 6, 0, 235))
            draw.text((65, 50), "MAKROKONTEXT", font=font_brand, fill="#FFFFFF")

            final_thumb = img.convert("RGB")
            final_thumb.save(raw_thumb_path, "JPEG", quality=95)
            logger.info("Profesyonel Thumbnail başarıyla mühürlendi!")
        except Exception as e:
            logger.warning(f"Pillow thumbnail tipografi uyarısı (orijinal korunuyor): {e}")

    async def _generate_single_tts_and_srt(self, text: str, mp3_path: Path, srt_path: Path, retries: int = 4):
        if mp3_path.exists() and srt_path.exists() and mp3_path.stat().st_size > 1000:
            return

        clean_text = " ".join(text.strip().split())
        voices = [self.voice] + self.fallback_voices

        for attempt in range(1, retries + 1):
            active_voice = voices[(attempt - 1) % len(voices)]
            try:
                communicate = edge_tts.Communicate(text=clean_text, voice=active_voice)
                submaker = edge_tts.SubMaker()

                with open(mp3_path, "wb") as audio_file:
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            audio_file.write(chunk["data"])
                        elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                            submaker.feed(chunk)

                if not mp3_path.exists() or mp3_path.stat().st_size < 500:
                    raise RuntimeError("Boş ses dosyası.")

                with open(srt_path, "w", encoding="utf-8") as srt_file:
                    srt_file.write(submaker.get_srt())
                return
            except Exception as e:
                logger.warning(f"Edge-TTS deneme {attempt}/{retries} ({active_voice}): {e}")
                mp3_path.unlink(missing_ok=True)
                if attempt < retries:
                    await asyncio.sleep(2 * attempt)
                else:
                    raise RuntimeError(f"Edge-TTS ses üretemedi: {mp3_path.name}")

    async def generate_all_audio(self):
        for scene in self.plan.scenes:
            idx = scene.scene_number
            await self._generate_single_tts_and_srt(
                scene.narration_text,
                self.audio_dir / f"scene_{idx}.mp3",
                self.audio_dir / f"scene_{idx}.srt"
            )
            await asyncio.sleep(1.2)

    def generate_all_visuals(self):
        base_seed = int(time.time()) % 10000
        for scene in self.plan.scenes:
            idx = scene.scene_number
            img_path = self.scenes_dir / f"scene_{idx}.jpg"
            need_sleep = not img_path.exists()
            self.download_safe(scene.visual_prompt_en, img_path, base_seed + idx)
            if need_sleep:
                time.sleep(3)

        thumb_path = OUTPUT_DIR / "thumbnail.jpg"
        self.download_safe(self.plan.seo.thumbnail_prompt_en, thumb_path, base_seed + 99)
        self.generate_professional_thumbnail(thumb_path, self.plan.seo.youtube_title)

    def run_pipeline(self):
        logger.info(f"=== Medya Üretimi Başlıyor (Hedef Ses: {self.voice}) ===")
        asyncio.run(self.generate_all_audio())
        self.generate_all_visuals()
        logger.info("=== Tüm Görsel, Ses, Altyazı ve Tipografik Thumbnail Hazır! ===")


if __name__ == "__main__":
    plan_file = OUTPUT_DIR / "latest_plan.json"
    if not plan_file.exists():
        raise FileNotFoundError("latest_plan.json bulunamadı!")
    plan_obj = ProductionPlanSchema.model_validate(json.loads(plan_file.read_text(encoding="utf-8")))
    MediaGenerator(plan_obj).run_pipeline()
