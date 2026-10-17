import json
import streamlit as st
from pathlib import Path

from src.config import OUTPUT_DIR, HISTORY_FILE, LANGUAGE_CONFIG, TARGET_LANGUAGE
from src.agent_researcher import NewsIntelligenceAgent, SelectedTopicSchema, TopicMemory
from src.agents_editorial import EditorialProductionCrew, ProductionPlanSchema
from src.media_generator import MediaGenerator
from src.video_renderer import VideoRenderer

st.set_page_config(
    page_title="AI Documentary Studio | Command Center",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Otonom YouTube Belgesel Stüdyosu — Komuta Merkezi")
st.markdown("16:9 Sinematik Haber & Analiz Belgeselleri | Multi-Agent Mimari | UrhG & YPP Uyumlu")

# ==========================================
# YAN MENÜ (SIDEBAR) - KONTROLLER VE TETİKLEYİCİLER
# ==========================================
with st.sidebar:
    st.header("⚙️ Sistem Parametreleri")
    selected_lang = st.selectbox(
        "Hedef Pazar & Dil (RPM Odaklı)",
        options=list(LANGUAGE_CONFIG.keys()),
        index=list(LANGUAGE_CONFIG.keys()).index(TARGET_LANGUAGE if TARGET_LANGUAGE in LANGUAGE_CONFIG else "DE"),
        format_func=lambda x: f"{x} - {LANGUAGE_CONFIG[x]['name']}"
    )
    st.caption(f"Aktif Ses Modeli: `{LANGUAGE_CONFIG[selected_lang]['voice']}`")
    st.divider()

    st.header("🚀 Otonom İş Akışı Tetikleyicileri")

    # 1. Ajan 1 Butonu
    if st.button("1️⃣ Ajan 1: Haber Tara & Konu Seç", use_container_width=True):
        with st.spinner("RSS akışları taranıyor ve en yüksek potansiyelli konu seçiliyor..."):
            try:
                agent1 = NewsIntelligenceAgent(language=selected_lang)
                topic = agent1.select_best_topic(auto_save_to_memory=True)
                st.success(f"Seçildi: {topic.topic_id} (Skor: {topic.viral_score})")
            except Exception as e:
                st.error(f"Ajan 1 Hatası: {e}")

    # 2. Ajan 2-3-4 Butonu
    if st.button("2️⃣ Ajan 2-3-4: Senaryo & Plan Üret", use_container_width=True):
        with st.spinner("Başyazar, Görsel Direktörü ve Moderatör prodüksiyon planını hazırlıyor..."):
            try:
                memory = TopicMemory()
                records = memory.load_and_prune()
                if not records:
                    st.warning("Hafızada konu yok! Önce Ajan 1'i çalıştırın.")
                else:
                    last = records[-1]
                    topic_obj = SelectedTopicSchema(
                        topic_id=last["topic_id"],
                        original_headline=last["original_headline"],
                        source_url=last["source_url"],
                        core_event_summary=last["core_event_summary"],
                        editorial_angle="Tiefgehende strukturelle, technologische und ökonomische Analyse.",
                        viral_score=85
                    )
                    crew = EditorialProductionCrew(language=selected_lang)
                    # Yeni plan üretildiğinde eski sahne/ses dosyalarını temizle
                    for old_file in (OUTPUT_DIR / "scenes").glob("*.jpg"):
                        old_file.unlink(missing_ok=True)
                    for old_file in (OUTPUT_DIR / "audio").glob("*.*"):
                        old_file.unlink(missing_ok=True)
                    crew.create_production_plan(topic_obj, save_to_disk=True)
                    st.success("Prodüksiyon planı (latest_plan.json) üretildi!")
            except Exception as e:
                st.error(f"Ajan 2-3-4 Hatası: {e}")

    # 3. Aşama 3: Medya Üretim Butonu
    if st.button("3️⃣ Aşama 3: Görsel & Ses/SRT İndir", use_container_width=True):
        plan_file = OUTPUT_DIR / "latest_plan.json"
        if not plan_file.exists():
            st.warning("Önce prodüksiyon planı üretmelisiniz!")
        else:
            with st.spinner("Pollinations 16:9 görselleri ve Edge-TTS ses/altyazıları üretiliyor..."):
                try:
                    plan_obj = ProductionPlanSchema.model_validate(json.loads(plan_file.read_text(encoding="utf-8")))
                    gen = MediaGenerator(plan_obj)
                    gen.run_pipeline()
                    st.success("Tüm görsel, ses ve .srt dosyaları hazır!")
                except Exception as e:
                    st.error(f"Medya Üretim Hatası: {e}")

    # 4. Aşama 4: FFmpeg Render Butonu
    if st.button("4️⃣ Aşama 4: FFmpeg 1080p Render", use_container_width=True):
        plan_file = OUTPUT_DIR / "latest_plan.json"
        if not plan_file.exists():
            st.warning("Önce prodüksiyon planı ve medyaları üretmelisiniz!")
        else:
            with st.spinner("FFmpeg Ken Burns (4K->1080p), -24dB miksaj ve altyazı gömme yapılıyor..."):
                try:
                    plan_obj = ProductionPlanSchema.model_validate(json.loads(plan_file.read_text(encoding="utf-8")))
                    renderer = VideoRenderer(plan_obj)
                    video_path = renderer.render_full_video()
                    st.success(f"Render Tamamlandı: {video_path.name}")
                except Exception as e:
                    st.error(f"Render Hatası: {e}")

    st.divider()
    if st.button("🔥 TAM OTONOM PIPELINE (1 -> 4)", type="primary", use_container_width=True):
        with st.status("Tam otonom belgesel üretimi çalışıyor...", expanded=True) as status:
            try:
                st.write("🔍 Ajan 1: Güncel haberler taranıyor...")
                agent1 = NewsIntelligenceAgent(language=selected_lang)
                topic = agent1.select_best_topic(auto_save_to_memory=True)

                st.write("✍️ Ajan 2-3-4: Senaryo, Flux promptları ve SEO kurgulanıyor...")
                for old_file in (OUTPUT_DIR / "scenes").glob("*.jpg"):
                    old_file.unlink(missing_ok=True)
                for old_file in (OUTPUT_DIR / "audio").glob("*.*"):
                    old_file.unlink(missing_ok=True)
                crew = EditorialProductionCrew(language=selected_lang)
                plan_obj = crew.create_production_plan(topic, save_to_disk=True)

                st.write("🎨 Aşama 3: 16:9 görseller ve seslendirmeler üretiliyor...")
                gen = MediaGenerator(plan_obj)
                gen.run_pipeline()

                st.write("🎞️ Aşama 4: FFmpeg 1080p sinematik render alınıyor...")
                renderer = VideoRenderer(plan_obj)
                renderer.render_full_video()

                status.update(label="✅ Belgesel Üretimi Başarıyla Tamamlandı!", state="complete", expanded=False)
            except Exception as e:
                status.update(label=f"❌ Hata: {e}", state="error")

# ==========================================
# ANA EKRAN SEKME YAPISI (TABS)
# ==========================================
tab1, tab2, tab3 = st.tabs([
    "📋 Aktif Prodüksiyon Planı & Sahneler",
    "🎥 Final Video & SEO Çıktısı",
    "🧠 30 Günlük Konu Hafızası (history.json)"
])

plan_path = OUTPUT_DIR / "latest_plan.json"

with tab1:
    if plan_path.exists():
        plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Konu ID", plan_data.get("topic_id", "-"))
        col2.metric("Dil", plan_data.get("language", "-"))
        col3.metric("Sahne Sayısı", len(plan_data.get("scenes", [])))
        col4.metric("Fon Müziği (-24 dB)", plan_data.get("supervision", {}).get("background_music", "-"))

        st.info(f"**🎯 İlk 5 Saniye Kancası (Hook):** {plan_data.get('hook_summary')}")
        st.success(f"**⚖️ Hukuki & YPP Denetim Notu (Ajan 4):** {plan_data.get('supervision', {}).get('compliance_notes')}")

        st.subheader("🎞️ Sahne Akışı (Storyboard)")
        for scene in plan_data.get("scenes", []):
            s_num = scene["scene_number"]
            with st.expander(f"Sahne {s_num} | Kamera Efekti: {scene['camera_effect']}", expanded=(s_num == 1)):
                c_img, c_txt = st.columns([1, 1.3])
                img_file = OUTPUT_DIR / "scenes" / f"scene_{s_num}.jpg"
                aud_file = OUTPUT_DIR / "audio" / f"scene_{s_num}.mp3"

                with c_img:
                    if img_file.exists():
                        st.image(str(img_file), caption=f"scene_{s_num}.jpg (16:9)", use_container_width=True)
                    else:
                        st.warning("Görsel henüz indirilmedi.")
                with c_txt:
                    st.markdown(f"**🎙️ Seslendirme Metni:**\n> {scene['narration_text']}")
                    st.caption(f"**🎨 Pollinations Flux Prompt:** `{scene['visual_prompt_en']}`")
                    if aud_file.exists():
                        st.audio(str(aud_file))
    else:
        st.info("Henüz `output/latest_plan.json` bulunamadı. Sol menüden Ajan 1 ve 2'yi çalıştırın.")

with tab2:
    final_vid = OUTPUT_DIR / "final_documentary.mp4"
    thumb_img = OUTPUT_DIR / "thumbnail.jpg"
    c_vid, c_seo = st.columns([1.3, 1])

    with c_vid:
        st.subheader("🎬 Render Edilmiş Final Belgesel")
        if final_vid.exists():
            st.video(str(final_vid))
        else:
            st.warning("`final_documentary.mp4` henüz oluşturulmadı. Sol menüden 'Aşama 4: FFmpeg Render' butonuna basın.")

        if thumb_img.exists():
            st.subheader("🖼️ Otomatik Üretilen YouTube Thumbnail (16:9)")
            st.image(str(thumb_img), use_container_width=True)

    with c_seo:
        st.subheader("📈 YouTube SEO & Metadata (Ajan 4)")
        if plan_path.exists():
            seo = json.loads(plan_path.read_text(encoding="utf-8")).get("seo", {})
            st.text_input("YouTube Başlığı (CTR Odaklı)", value=seo.get("youtube_title", ""))
            st.text_area("YouTube Açıklaması", value=seo.get("youtube_description", ""), height=180)
            st.text_input("Etiketler (Tags)", value=", ".join(seo.get("tags", [])))
            st.caption(f"**Thumbnail Prompt:** `{seo.get('thumbnail_prompt_en', '')}`")

with tab3:
    st.subheader("🧠 İşlenen Konular Hafızası (Son 30 Gün)")
    if HISTORY_FILE.exists():
        history_data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        st.write(f"Toplam Kayıtlı Konu: **{len(history_data)}**")
        st.dataframe(history_data, use_container_width=True)
    else:
        st.info("Hafıza dosyası henüz boş.")
