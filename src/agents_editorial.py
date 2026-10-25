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
logger = logging.getLogger("Agents_2_3_4_CinematicCrew")


class SceneSchema(BaseModel):
    scene_number: int = Field(description="Sahne sırası (1'den 7'ye kadar).")
    act_title: str = Field(description="Sahnenin dramatik perde adı (Örn: Act I - The Classified File).")
    narration_text: str = Field(
        description="Bu sahnede seslendirilecek belgesel metni (Hedef dilde, derin, esrarengiz, sinematik bir hikaye anlatımıyla 45-65 kelime arası. Kısaltma kullanma)."
    )
    visual_prompts_en: List[str] = Field(
        min_length=3,
        max_length=3,
        description="Bu sahne boyunca her 5 saniyede bir değişecek TAM 3 ADET birbirinden farklı açıya sahip İNGİLİZCE sinematik görsel komutu (1: Geniş atmosferik plan, 2: Tarihi belge/nesne/detay yakın planı, 3: Gölgeli dramatik portre veya mekan). ASLA yazı/metin içermemeli."
    )
    camera_effects: List[Literal["zoom_in", "zoom_out", "pan_left_to_right", "pan_right_to_left"]] = Field(
        min_length=3,
        max_length=3,
        description="3 görselin her biri için sırasıyla kamera hareketi."
    )


class YouTubeSEOSchema(BaseModel):
    youtube_title: str = Field(
        description="Hedef dilde, gizem ve merak uyandıran, yıllarca tıklanacak (Evergreen) belgesel başlığı. İki nokta üst üsteden (:) önceki ilk 2-3 kelime kapak fotoğrafına dev harflerle basılacaktır (Örn: GEHEIMAKTE TESLA: Was das FBI 1943 wirklich beschlagnahmte)."
    )
    youtube_description: str = Field(description="Hedef dilde, zaman damgalı, derinlikli SEO açıklaması ve hashtagler.")
    tags: List[str] = Field(description="10-15 adet niş tarih, gizem, bilim ve analiz etiketi.")
    thumbnail_prompt_en: str = Field(
        description="YouTube kapağı için İNGİLİZCE ultra çarpıcı, chiaroscuro ışıklandırmalı, gizemli sinematik arka plan komutu (yarı karanlık portre, gizli dosya mührü, dramatik kırmızı/altın/karanlık kontrast). Alt kısım tipografi için koyu olmalı."
    )


class ComplianceAndAudioSchema(BaseModel):
    background_music: Literal["cinematic_tension.mp3", "serious_news.mp3", "ambient_tech.mp3"] = Field(description="Arka plan müziği")
    music_reasoning: str = Field(description="Müzik seçimi nedeni.")
    urhg_and_ypp_compliant: bool = Field(description="Hukuki uyum.")
    compliance_notes: str = Field(description="Denetim notu.")


class ProductionPlanSchema(BaseModel):
    topic_id: str
    language: str
    hook_summary: str = Field(description="İlk 5 saniye kancası.")
    scenes: List[SceneSchema] = Field(min_length=7, max_length=8)
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
        Sen 'MakroKontext' Sinematik Gizem & Tarih Stüdyosu'nun Prodüksiyon Ekibisin:
        1. AJAN 2 (Başyazar - Noir & Mystery): Lemmino, Fern ve Arte tarzında izleyiciyi ilk saniyeden ekrana kilitleyen, karanlık, entelektüel ve sürükleyici bir belgesel metni yazarsın. Dil: {self.config['name']} ({self.language}).
           - Anlatımı 7 sahnelik dramatik bir ark olarak kurgula: 1. Çarpıcı Sır (Hook), 2. Resmi Tarih, 3. Gizlenen Belgeler, 4. Gölgedeki Aktörler, 5. Kırılma Noktası (Twist), 6. Günümüze Yansıyan Güç, 7. Çözümsüz Miras (Final).
        2. AJAN 3 (Sinematograf): Tekdüze slayt gösterisi YASAKTIR! Her sahne için TAM 3 ADET farklı sinematik çekim açısı ('visual_prompts_en') yazacaksın (Toplam 21-24 görsel).
           - Her görsel promptuna şu sanatsal stili yansıt: 'cinematic 35mm film still, dark moody chiaroscuro lighting, vintage archival atmosphere, dramatic shadows, photorealistic 8k, no text'.
        3. AJAN 4 (SEO & Kapak Direktörü): Merak uyandıran, tıklama garantili bir kapak ve başlık kurgula. Başlığın ':' işaretinden önceki ilk 2-3 kelimesi kapakta dev sarı/kırmızı harflerle yazılacaktır (Örn: 'VERBOTENE AKTE', 'DAS GEHEIME PROTOKOLL', 'DER SCHATTEN-PAKT').
        """

        prompt = f"""
        KONU ID: {topic.topic_id}
        İLHAM ALINAN GÜNCEL OLAY: {topic.original_headline}
        TARİHSEL SIR / GİZEM KANCASI: {topic.historical_or_mystery_hook}
        ARŞİV VE OLGU ÖZETİ: {topic.core_event_summary}
        ANLATIM AÇISI: {topic.editorial_angle}
        HEDEF DİL: {self.language}
        """

        logger.info(f"Sinematik Senaryo ve 21+ Kesitli Görsel Plan üretiliyor: '{topic.topic_id}'...")
        response = generate_content_with_fallback(
            client=self.client,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=ProductionPlanSchema,
                temperature=0.45,
            )
        )

        plan = ProductionPlanSchema.model_validate_json(response.text)
        
        if save_to_disk:
            plan_path = OUTPUT_DIR / "latest_plan.json"
            with open(plan_path, "w", encoding="utf-8") as f:
                json.dump(plan.model_dump(), f, ensure_ascii=False, indent=2)
            logger.info(f"Sinematik Plan diske kaydedildi ({len(plan.scenes)*3} görsel kesiti): {plan_path}")

        return plan
