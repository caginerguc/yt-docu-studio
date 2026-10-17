import json
import logging
from typing import List, Literal
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    TARGET_LANGUAGE,
    OUTPUT_DIR,
    HISTORY_FILE,
    get_active_config
)
from src.agent_researcher import SelectedTopicSchema

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Agents_2_3_4_Editorial")


# ==========================================
# PYDANTIC ŞEMALARI (AJAN 2, 3 ve 4 ÇIKTISI)
# ==========================================
class SceneSchema(BaseModel):
    scene_number: int = Field(description="Sahne sırası (1'den başlar, toplam 6 ila 8 sahne).")
    narration_text: str = Field(
        description="Bu sahnede seslendirilecek belgesel metni (Hedef dilde, doğal, akıcı, 40-60 kelime arası). Kısaltma veya okunması zor semboller kullanma."
    )
    visual_prompt_en: str = Field(
        description="Pollinations Flux modeli için 16:9 geniş ekran İNGİLİZCE görsel üretim komutu. Fotogerçekçi, sinematik ışıklandırma, belgesel fotoğrafçılığı tarzında. ASLA metin, yazı, harf, logo veya gerçek politikacı yüzü içermemeli (Örn: 'Cinematic wide shot of Frankfurt banking district at dusk, glass skyscrapers reflecting stormy clouds, moody lighting, photorealistic, 8k, documentary style')."
    )
    camera_effect: Literal["zoom_in", "zoom_out", "pan_left_to_right", "pan_right_to_left"] = Field(
        description="FFmpeg Ken Burns sinematik kamera hareketi yönü."
    )


class YouTubeSEOSchema(BaseModel):
    youtube_title: str = Field(
        description="Hedef dilde, merak uyandıran, yüksek tıklama oranlı (CTR) ama clickbait olmayan belgesel başlığı (Maksimum 70 karakter)."
    )
    youtube_description: str = Field(
        description="Hedef dilde, SEO uyumlu, olayın arka planını özetleyen 2 paragraflık YouTube video açıklaması ve 3-5 adet #hashtag."
    )
    tags: List[str] = Field(
        description="Hedef dilde 8-12 adet yüksek aranma hacimli YouTube etiketi (virgülsüz liste)."
    )
    thumbnail_prompt_en: str = Field(
        description="YouTube kapak fotoğrafı (Thumbnail) için yüksek kontrastlı, çarpıcı, 16:9 İNGİLİZCE Flux görsel komutu (Yazı/metin içermemeli)."
    )


class ComplianceAndAudioSchema(BaseModel):
    background_music: Literal["ambient_tech.mp3", "serious_news.mp3", "cinematic_tension.mp3", "none"] = Field(
        description="Haberin atmosferine en uygun fon müziği seçimi. Çok hassas/trajik bir konuysa 'none' seçilmeli."
    )
    music_reasoning: str = Field(description="Neden bu fon müziğinin (veya sessizliğin) seçildiğinin kısa açıklaması.")
    urhg_and_ypp_compliant: bool = Field(
        description="Metin tamamen özgün mü (Leistungsschutzrecht uyumu) ve YouTube Reused Content / nefret söylemi kurallarına %100 uygun mu?"
    )
    compliance_notes: str = Field(description="Hukuki ve editoryal denetim notu.")


class ProductionPlanSchema(BaseModel):
    topic_id: str
    language: str
    hook_summary: str = Field(description="İlk 5 saniyede izleyiciyi videoda tutacak kanca stratejisi.")
    scenes: List[SceneSchema] = Field(
        description="6 ila 8 sahneden oluşan, toplamda 300-400 kelimelik (2.5 - 3 dakika) belgesel akışı.",
        min_length=6,
        max_length=8
    )
    seo: YouTubeSEOSchema
    supervision: ComplianceAndAudioSchema

    @property
    def full_script(self) -> str:
        """Tüm sahnelerin seslendirme metnini birleştirir."""
        return " ".join([s.narration_text for s in self.scenes])


# ==========================================
# AJAN 2, 3, 4: EDİTORYAL PRODÜKSİYON MOTORU
# ==========================================
class EditorialProductionCrew:
    def __init__(self, language: str = TARGET_LANGUAGE):
        if not GEMINI_API_KEY:
            raise EnvironmentError("GEMINI_API_KEY bulunamadı!")
        self.language = language.upper()
        self.config = get_active_config(self.language)
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def create_production_plan(self, topic: SelectedTopicSchema, save_to_disk: bool = True) -> ProductionPlanSchema:
        """Seçilen konuyu 3 ajanın perspektifiyle tam bir prodüksiyon planına dönüştürür."""
        
        system_instruction = f"""
        Sen 3 Kıdemli Uzmandan oluşan Otonom Bir Belgesel Prodüksiyon Ekibisin:
        1. AJAN 2 (Başyazar - Editorial Scriptwriter): DW Documentary, Arte veya ColdFusion kalitesinde metin yazarsın. Haber spikeri gibi sıkıcı okumazsın; analitik bir hikaye anlatıcısısındır.
           - Sahne 1 mutlaka sarsıcı bir 'Hook' (Kanca) ile başlamalıdır ("Herkes X'e odaklanırken, perde arkasında Y değişiyor...").
           - Orta sahneler olayın ekonomik, teknolojik ve yapısal arka planını derinlemesine inceler.
           - Son sahne gelecek projeksiyonu yapar ve izleyiciye düşündürücü bir soru sorarak yorumlara yönlendirir (CTA).
           - Metin dili: {self.config['name']} ({self.language}). TTS (Seslendirme) motorunun kusursuz okuması için sayıları ve kısaltmaları okunuşuyla veya açık yaz (Örn: 'EZB' yerine ilk geçtiği yerde 'Europäische Zentralbank').

        2. AJAN 3 (Görsel Direktörü - Cinematography Agent): Metni 6 ila 8 sahneye bölersin.
           - Her sahne için Pollinations.ai Flux modelinde üretilecek İNGİLİZCE ('visual_prompt_en') 16:9 görsel komutları yazarsın.
           - Görsellerde ASLA yazı, tabela metni, ekran yazısı veya gerçek yaşayan ünlü/politikacı portresi (Deepfake/Telif riski) kullanma. Sembolik, atmosferik, fotogerçekçi mimari, endüstriyel veya kavramsal sinematik sahneler betimle.
           - Ardışık sahnelerde aynı kamera efektini ('camera_effect') tekrarlama; dinamizm için çeşitlendir.

        3. AJAN 4 (Moderatör, Ses & SEO Süpervizörü):
           - Almanya Telif Hukuku (UrhG / Leistungsschutzrecht) uyarınca metnin kaynak haberden kopyalanmadığını, %100 özgün bir editoryal analiz (Transformative Content) olduğunu denetle.
           - Haberin tonuna göre 'ambient_tech.mp3' (teknoloji/bilim), 'serious_news.mp3' (ekonomi/finans/politika), 'cinematic_tension.mp3' (kriz/jeopolitik gerilim) veya 'none' seç.
           - Yüksek tıklama oranlı (CTR) {self.language} YouTube başlığı ve açıklaması üret.
        """

        prompt = f"""
        Aşağıdaki haber olgusunu ve editoryal açıyı kullanarak 6 ila 8 sahnelik eksiksiz belgesel prodüksiyon planını oluştur:

        KONU ID: {topic.topic_id}
        ORİJİNAL BAŞLIK: {topic.original_headline}
        OLGUSAL ÖZET: {topic.core_event_summary}
        EDİTORYAL ANALİZ AÇISI: {topic.editorial_angle}
        HEDEF DİL: {self.language}
        """

        logger.info(f"Ajan 2, 3 ve 4 çalışıyor: '{topic.topic_id}' için senaryo ve görsel plan üretiliyor...")
        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=ProductionPlanSchema,
                temperature=0.4,
            ),
        )

        plan = ProductionPlanSchema.model_validate_json(response.text)
        total_words = len(plan.full_script.split())
        logger.info(
            f"Prodüksiyon Planı Hazır! Sahne Sayısı: {len(plan.scenes)} | "
            f"Toplam Kelime: {total_words} | Müzik: {plan.supervision.background_music}"
        )

        if save_to_disk:
            plan_path = OUTPUT_DIR / "latest_plan.json"
            with open(plan_path, "w", encoding="utf-8") as f:
                json.dump(plan.model_dump(), f, ensure_ascii=False, indent=2)
            logger.info(f"Plan diske kaydedildi: {plan_path}")

        return plan


# ==========================================
# MODÜL TEST BLOĞU
# ==========================================
if __name__ == "__main__":
    print("--- Ajan 2, 3 ve 4 Testi Başlatılıyor ---")
    # Hafızadaki son seçilen konuyu oku (API kotası harcamadan Ajan 1'in çıktısını kullan)
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
    
    if not history:
        raise RuntimeError("history.json boş! Önce Ajan 1'i çalıştırın.")

    last_entry = history[-1]
    mock_topic = SelectedTopicSchema(
        topic_id=last_entry["topic_id"],
        original_headline=last_entry["original_headline"],
        source_url=last_entry["source_url"],
        core_event_summary=last_entry["core_event_summary"],
        editorial_angle="Tiefgehende Analyse der strukturellen und wirtschaftlichen Auswirkungen für Europa.",
        viral_score=88
    )

    crew = EditorialProductionCrew(language=last_entry.get("language", TARGET_LANGUAGE))
    production_plan = crew.create_production_plan(mock_topic, save_to_disk=True)

    print(f"\n[BAŞARILI] YouTube Başlığı: {production_plan.seo.youtube_title}")
    print(f"Seçilen Müzik: {production_plan.supervision.background_music} ({production_plan.supervision.music_reasoning})")
    print(f"Toplam Sahne: {len(production_plan.scenes)} | Toplam Kelime: {len(production_plan.full_script.split())}\n")
    for s in production_plan.scenes[:2]:
        print(f"--- Sahne {s.scene_number} ({s.camera_effect}) ---")
        print(f"Metin ({production_plan.language}): {s.narration_text}")
        print(f"Flux Prompt (EN): {s.visual_prompt_en}\n")
