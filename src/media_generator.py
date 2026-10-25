import asyncio
import json
import logging
import time
from pathlib import Path
from urllib.parse import quote
import requests
import edge_tts
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

from src.config import OUTPUT_DIR, get_active_config, POLLINATIONS_API_KEY
from src.agents_editorial import ProductionPlanSchema

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Stage3_CinematicMedia")


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

    def download_image_pollinations(self, prompt: str, output_path: Path, seed: int = 42) -> Path:
        if output_path.exists() and output_path.stat().st_size > 10000:
            return output_path

        style_suffix = "cinematic 35mm film grain, dark moody chiaroscuro lighting, dramatic shadows, archival documentary style, 16:9 widescreen, 8k, no text, no watermark"
        enhanced_prompt = f"{prompt}, {style_suffix}"
        encoded_prompt = quote(enhanced_prompt)
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        if self.pollinations_token:
            headers["Authorization"] = f"Bearer {self.pollinations_token}"

        for model_name in ["flux", "turbo"]:
            for attempt in range(1, 3):
                current_seed = seed + (attempt * 13)
                url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1920&height=1080&seed={current_seed}&model={model_name}&nologo=true"
                if self.pollinations_token:
                    url += f"&key={self.pollinations_token}"

                try:
                    logger.info(f"Çekim indiriliyor ({model_name}, deneme {attempt}): {output_path.name}...")
                    response = requests.get(url, timeout=75, headers=headers)
                    if response.status_code == 200 and len(response.content) > 8000:
                        with open(output_path, "wb") as f:
                            f.write(response.content)
                        return output_path
                except Exception as e:
                    logger.warning(f"Pollinations uyarısı ({model_name}): {e}")
                time.sleep(3 * attempt)

        raise RuntimeError("Pollinations tüm denemelerde başarısız oldu.")

    def create_studio_fallback_canvas(self, output_path: Path):
        img = Image.new("RGB", (1920, 1080), (10, 12, 18))
        draw = ImageDraw.Draw(img)
        for y in range(1080):
            c = int(8 + (y / 1080) * 20)
            draw.line([(0, y), (1920, y)], fill=(c, c + 2, c + 8))
        img.save(output_path, "JPEG", quality=92)
        return output_path

    def download_safe(self, prompt: str, output_path: Path, seed: int = 42):
        try:
            self.download_image_pollinations(prompt, output_path, seed)
        except Exception as e:
            logger.warning(f"Görsel API uyarısı ({e}). Yedek karanlık tuval üretiliyor...")
            self.create_studio_fallback_canvas(output_path)

    def generate_professional_thumbnail(self, raw_thumb_path: Path, title: str):
        try:
            logger.info("Pillow ile Sinematik Gizem Thumbnail'i işleniyor...")
            img = Image.open(raw_thumb_path).convert("RGBA")
            img = img.resize((1280, 720), Image.Resampling.LANCZOS)
            
            img = ImageEnhance.Contrast(img).enhance(1.35)
            img = ImageEnhance.Color(img).enhance(0.85)  # Hafif desatüre sinematik noir tonu
            
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw_overlay = ImageDraw.Draw(overlay)
            for y in range(320, 720):
                alpha = int(215 * ((y - 320) / 400))
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
                    font_title = ImageFont.truetype(font_name, 74)
                    font_brand = ImageFont.truetype(font_name, 30)
                    break
                except Exception:
                    continue
            
            if not font_title:
                font_title = ImageFont.load_default()
                font_brand = ImageFont.load_default()

            clean_title = title.split(":")[0].replace("?", "").replace("!", "").strip().upper()
            words = clean_title.split()
            display_text = " ".join(words[:4]) if len(words) > 4 else clean_title

            # Kırmızı "STRENG GEHEIM / MAKROKONTEXT" üst mühür bandı
            draw.rectangle([(48, 38), (360, 88)], fill=(195, 0, 0, 240))
            draw.text((62, 47), "MAKROKONTEXT", font=font_brand, fill="#FFFFFF")

            # Alt vurucu sarı/siyah dev başlık
            draw.text((48, 555), display_text, font=font_title, fill="#FFE600", stroke_width=7, stroke_fill="#000000")

            img.convert("RGB").save(raw_thumb_path, "JPEG", quality=95)
        except Exception as e:
            logger.warning(f"Thumbnail tipografi uyarısı: {e}")

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
                logger.warning(f"Edge-TTS deneme {attempt}/{retries}: {e}")
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
        # Eski sahne görsellerini temizle ki önceki videonun kareleri karışmasın
        for old_img in self.scenes_dir.glob("*.jpg"):
            old_img.unlink(missing_ok=True)

        base_seed = int(time.time()) % 10000
        for scene in self.plan.scenes:
            idx = scene.scene_number
            prompts = getattr(scene, "visual_prompts_en", None) or [getattr(scene, "visual_prompt_en", "")]
            for sub_idx, prompt in enumerate(prompts, start=1):
                img_path = self.scenes_dir / f"scene_{idx}_{sub_idx}.jpg"
                self.download_safe(prompt, img_path, base_seed + (idx * 10) + sub_idx)
                time.sleep(2.5)

        thumb_path = OUTPUT_DIR / "thumbnail.jpg"
        thumb_path.unlink(missing_ok=True)
        self.download_safe(self.plan.seo.thumbnail_prompt_en, thumb_path, base_seed + 999)
        self.generate_professional_thumbnail(thumb_path, self.plan.seo.youtube_title)

    def run_pipeline(self):
        logger.info(f"=== V3.0 Sinematik Medya Üretimi Başlıyor (Ses: {self.voice}) ===")
        asyncio.run(self.generate_all_audio())
        self.generate_all_visuals()
        logger.info("=== 21+ Sinematik Çekim, Ses ve Noir Kapak Hazır! ===")
