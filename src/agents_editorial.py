import json
import logging
from typing import List, Literal
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from src.config import (
    GEMINI_API_KEY,
    TARGET_LANGUAGE,
    OUTPUT_DIR,
    get_active_config,
    generate_content_with_fallback
)
from src.agent_researcher import SelectedTopicSchema

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Agents_2_3_4_Editorial")

class SceneSchema(BaseModel):
    scene_number: int = Field(description="Sahne sırası (1'den başlar, toplam 6 ila 8 sahne).")
    narration_text: str = Field(description="Bu sahnede seslendirilecek belgesel metni (Hedef dilde, doğal, akıcı, 40-60 kelime arası).")
    visual_prompt_en: str = Field(description="Pollinations Flux modeli için 16:9 geniş ekran İNGİLİZCE görsel komutu. ASLA metin, yazı, harf, logo içermemeli.")
    camera_effect: Literal["zoom_in", "zoom_out", "pan_left_to_right", "pan_right_to_left"] = Field(description="Kamera hareketi.")

class YouTubeSEOSchema(BaseModel):
    youtube_title: str = Field(description="Hedef dilde, merak uyandıran, yüksek tıklama oranlı (CTR) belgesel başlığı. İlk kısmı iki nokta üst üste (:) ile ayrılmış 2-3 kelimelik çok güçlü bir vurgu olmalı (Örn: CHINAS STROM-KRIEG: Wie Peking...).")
    youtube_description: str = Field(description="Hedef dilde, SEO uyumlu video açıklaması ve hashtagler.")
    tags: List[str] = Field(description="8-12 adet YouTube etiketi.")
    thumbnail_prompt_en: str = Field(
        description="YouTube kapak fotoğrafı arka planı için İNGİLİZCE yüksek kontrastlı, dramatik sinematik görsel komutu (bayraklar, endüstriyel tesisler, lider silüetleri, kırmızı/mavi dramatik ışıklandırma). Üzerine Pillow ile yazı basılacağı için alt kısmı nispeten koyu ve temiz olmalı."
    )

class ComplianceAndAudioSchema(BaseModel):
    background_music: Literal["ambient_tech.mp3", "serious_news.mp3", "cinematic_tension.mp3", "none"] = Field(description="Arka plan müziği")
    music_reasoning: str = Field(description="Müzik seçimi nedeni.")
    urhg_and_ypp_compliant: bool = Field(description="Hukuki uyum.")
    compliance_notes: str = Field(description="Denetim notu.")

class ProductionPlanSchema(BaseModel):
    topic_id: str
    language: str
    hook_summary: str = Field(description="İlk 5 saniye kancası.")
    scenes: List[SceneSchema] = Field(min_length=6, max_length=8)
    seo: YouTubeSEOSchema
    supervision: ComplianceAndAudioSchema

    @property
    def full_script(self) -> str:
        return " ".join([s.narration_text for s in self.scenes])

class EditorialProductionCrew:
    def __init__(self, language: str = TARGET_LANGUAGE):
        if not GEMINI_API_KEY:
            raise EnvironmentError("GEMINI_API_KEY bulunamadı!")
        self.language = language.upper()
        self.config = get_active_config(self.language)
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def create_production_plan(self, topic: SelectedTopicSchema, save_to_disk: bool = True) -> ProductionPlanSchema:
        system_instruction = f"""
        Sen 3 Kıdemli Uzmandan oluşan Otonom Bir Belgesel Prodüksiyon Ekibisin:
        1. AJAN 2 (Başyazar): DW Documentary, Arte veya ColdFusion kalitesinde analitik metin yazarsın. Dil: {self.config['name']} ({self.language}).
        2. AJAN 3 (Görsel Direktörü): Metni 6-8 sahneye böler, sahneler için (yazısız) sinematik Flux promptları yazarsın.
        3. AJAN 4 (Moderatör, Ses & SEO Süpervizörü):
           - Yüksek tıklama oranlı (CTR) {self.language} YouTube başlığı ve açıklaması üret. Başlığın iki nokta üst üsteden (:) önceki ilk 2-3 kelimesi kapak fotoğrafına dev harflerle basılacaktır, bu yüzden çok vurucu seç!
           - 'thumbnail_prompt_en' alanında, YouTube algoritmasında dikkat çekecek dramatik kontrastlı (kırmızı/mavi) sinematik bir arka plan sahnesi betimle.
        """

        prompt = f"""
        KONU ID: {topic.topic_id}
        ORİJİNAL BAŞLIK: {topic.original_headline}
        OLGUSAL ÖZET: {topic.core_event_summary}
        EDİTORYAL ANALİZ AÇISI: {topic.editorial_angle}
        HEDEF DİL: {self.language}
        """

        logger.info(f"Ajan 2, 3 ve 4 çalışıyor: '{topic.topic_id}' için senaryo ve görsel plan üretiliyor...")
        response = generate_content_with_fallback(
            client=self.client,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=ProductionPlanSchema,
                temperature=0.4,
            )
        )

        plan = ProductionPlanSchema.model_validate_json(response.text)
        
        if save_to_disk:
            plan_path = OUTPUT_DIR / "latest_plan.json"
            with open(plan_path, "w", encoding="utf-8") as f:
                json.dump(plan.model_dump(), f, ensure_ascii=False, indent=2)
            logger.info(f"Plan diske kaydedildi: {plan_path}")

        return plan
