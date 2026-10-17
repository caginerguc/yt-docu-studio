import json
import logging
import re
import subprocess
from pathlib import Path
from typing import List

from src.config import ASSETS_DIR, OUTPUT_DIR
from src.agents_editorial import ProductionPlanSchema

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Stage4_VideoRenderer")


class VideoRenderer:
    def __init__(self, plan: ProductionPlanSchema):
        self.plan = plan
        self.scenes_dir = OUTPUT_DIR / "scenes"
        self.audio_dir = OUTPUT_DIR / "audio"
        self.clips_dir = OUTPUT_DIR / "clips"
        self.music_dir = ASSETS_DIR / "music"
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.music_dir.mkdir(parents=True, exist_ok=True)
        self.fps = 30

    def get_audio_duration(self, audio_path: Path) -> float:
        """ffprobe ile ses dosyasının tam süresini (saniye) ölçer."""
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())

    def ensure_background_music(self, track_name: str) -> Path | None:
        """Seçilen fon müziği assets/music/ içinde yoksa FFmpeg ile telifsiz ambiyans/drone üretir."""
        if not track_name or track_name == "none":
            return None

        track_path = self.music_dir / track_name
        if track_path.exists() and track_path.stat().st_size > 10000:
            return track_path

        logger.info(f"'{track_name}' bulunamadı. FFmpeg ile %100 telifsiz sinematik ambiyans sentezleniyor...")
        # Haberin moduna göre harmonik akor frekansları
        if "tech" in track_name:
            expr = "0.15*sin(2*PI*110*t)+0.10*sin(2*PI*164.81*t)+0.08*sin(2*PI*220*t)+0.05*sin(2*PI*329.63*t)"
        elif "tension" in track_name:
            expr = "0.18*sin(2*PI*65.41*t)+0.12*sin(2*PI*98.0*t)+0.08*sin(2*PI*138.59*t)"
        else:  # serious_news.mp3 (Derin D-Minor belgesel pad)
            expr = "0.15*sin(2*PI*73.42*t)+0.12*sin(2*PI*110.0*t)+0.08*sin(2*PI*146.83*t)+0.05*sin(2*PI*220.0*t)"

        cmd = [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"aevalsrc='{expr}:s=44100:d=300'",
            "-af", "lowpass=f=600,tremolo=f=0.25:d=0.4",
            "-c:a", "libmp3lame", "-b:a", "128k",
            str(track_path)
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        logger.info(f"Ambiyans fon müziği üretildi: {track_path.name}")
        return track_path

    def _build_zoompan_filter(self, effect: str, duration: float) -> str:
        """Titremeyi (jitter) önlemek için 4K upscale + 1080p Ken Burns zoompan filtresi."""
        frames = int(duration * self.fps) + 15
        if effect == "zoom_in":
            z_expr = f"min(zoom+0.0005,1.20)"
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = "ih/2-(ih/zoom/2)"
        elif effect == "zoom_out":
            z_expr = f"if(eq(on,1),1.20,max(1.001,zoom-0.0005))"
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = "ih/2-(ih/zoom/2)"
        elif effect == "pan_left_to_right":
            z_expr = "1.15"
            x_expr = f"(iw-iw/zoom)*(on/{frames})"
            y_expr = "ih/2-(ih/zoom/2)"
        else:  # pan_right_to_left
            z_expr = "1.15"
            x_expr = f"(iw-iw/zoom)*(1-on/{frames})"
            y_expr = "ih/2-(ih/zoom/2)"

        # Yumuşak sahne geçişi için her sahnenin başına ve sonuna 0.4 sn fade ekliyoruz
        fade_out_start = max(0.1, duration - 0.4)
        return (
            f"scale=3840:2160:force_original_aspect_ratio=increase,crop=3840:2160,"
            f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':d={frames}:s=1920x1080:fps={self.fps},"
            f"fade=t=in:st=0:d=0.4,fade=t=out:st={fade_out_start:.2f}:d=0.4,format=yuv420p"
        )

    @staticmethod
    def _srt_time_to_seconds(time_str: str) -> float:
        h, m, s_ms = time_str.split(":")
        s, ms = s_ms.replace(".", ",").split(",")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

    @staticmethod
    def _seconds_to_srt_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int(round((seconds - int(seconds)) * 1000))
        if ms >= 1000:
            ms = 999
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def build_master_srt(self, scene_durations: List[float]) -> Path:
        """Tüm sahnelerin bireysel .srt dosyalarını zaman ofsetleriyle tek bir master.srt'de birleştirir."""
        master_srt_path = OUTPUT_DIR / "master_subtitles.srt"
        combined_blocks = []
        global_counter = 1
        time_offset = 0.0

        for idx, duration in enumerate(scene_durations, start=1):
            srt_file = self.audio_dir / f"scene_{idx}.srt"
            if srt_file.exists():
                content = srt_file.read_text(encoding="utf-8").strip()
                blocks = re.split(r"\n\s*\n", content)
                for block in blocks:
                    lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
                    if len(lines) >= 2:
                        time_line_idx = 1 if "-->" in lines[1] else 0
                        if "-->" not in lines[time_line_idx]:
                            continue
                        start_str, end_str = [t.strip() for t in lines[time_line_idx].split("-->")]
                        new_start = self._seconds_to_srt_time(self._srt_time_to_seconds(start_str) + time_offset)
                        new_end = self._seconds_to_srt_time(self._srt_time_to_seconds(end_str) + time_offset)
                        text_lines = " ".join(lines[time_line_idx + 1:])
                        combined_blocks.append(f"{global_counter}\n{new_start} --> {new_end}\n{text_lines}\n")
                        global_counter += 1
            time_offset += duration

        master_srt_path.write_text("\n".join(combined_blocks), encoding="utf-8")
        logger.info(f"Birleştirilmiş altyazı hazır: {master_srt_path.name} ({global_counter - 1} satır)")
        return master_srt_path

    def render_full_video(self) -> Path:
        """Tüm sahneleri, -24 dB fon müziğini ve altyazıları birleştirip 1080p MP4 üretir."""
        logger.info("=== Aşama 4: FFmpeg Sinematik Render Başlıyor ===")
        clip_paths = []
        scene_durations = []

        # 1. Her sahneyi kendi ses süresine göre Ken Burns efektiyle klip haline getir
        for scene in self.plan.scenes:
            idx = scene.scene_number
            img_path = self.scenes_dir / f"scene_{idx}.jpg"
            aud_path = self.audio_dir / f"scene_{idx}.mp3"
            clip_path = self.clips_dir / f"clip_{idx}.mp4"

            duration = self.get_audio_duration(aud_path)
            scene_durations.append(duration)
            vf_filter = self._build_zoompan_filter(scene.camera_effect, duration)

            logger.info(f"Sahne {idx}/{len(self.plan.scenes)} render ediliyor ({duration:.1f} sn | Efekt: {scene.camera_effect})...")
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", str(img_path),
                "-i", str(aud_path),
                "-vf", vf_filter,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
                "-t", f"{duration:.3f}",
                str(clip_path)
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            clip_paths.append(clip_path)

        # 2. Tüm sahne kliplerini uç uca birleştir (Concat)
        concat_list_path = OUTPUT_DIR / "concat_list.txt"
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for cp in clip_paths:
                f.write(f"file '{cp.resolve().as_posix()}'\n")

        merged_raw_path = OUTPUT_DIR / "merged_raw.mp4"
        logger.info("Sahneler birleştiriliyor...")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list_path), "-c", "copy", str(merged_raw_path)
        ], capture_output=True, check=True)

        # 3. Master SRT altyazısını oluştur
        master_srt = self.build_master_srt(scene_durations)
        # Windows FFmpeg subtitles filtresi için dizin yolu kaçış (escape) formatı
        escaped_srt_path = master_srt.resolve().as_posix().replace(":", r"\:")

        # 4. Fon müziği (-24 dB = 0.063 genlik) ve Altyazı Gömme (Final Render)
        final_video_path = OUTPUT_DIR / "final_documentary.mp4"
        music_path = self.ensure_background_music(self.plan.supervision.background_music)

        sub_style = "FontName=Arial,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,BackColour=&H80000000,BorderStyle=4,Outline=1,Shadow=0,MarginV=35,Bold=1"
        vf_sub = f"subtitles='{escaped_srt_path}':force_style='{sub_style}'"

        logger.info("Final miksaj (-24 dB Audio Ducking) ve Altyazı gömme işlemi uygulanıyor...")
        if music_path and music_path.exists():
            cmd_final = [
                "ffmpeg", "-y",
                "-i", str(merged_raw_path),
                "-stream_loop", "-1", "-i", str(music_path),
                "-filter_complex", f"[1:a]volume=0.063[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout];[0:v]{vf_sub}[vout]",
                "-map", "[vout]", "-map", "[aout]",
                "-c:v", "libx264", "-preset", "fast", "-crf", "19",
                "-c:a", "aac", "-b:a", "192k",
                str(final_video_path)
            ]
        else:
            cmd_final = [
                "ffmpeg", "-y",
                "-i", str(merged_raw_path),
                "-vf", vf_sub,
                "-c:v", "libx264", "-preset", "fast", "-crf", "19",
                "-c:a", "copy",
                str(final_video_path)
            ]

        subprocess.run(cmd_final, capture_output=True, check=True)
        size_mb = final_video_path.stat().st_size / (1024 * 1024)
        total_time = sum(scene_durations)
        logger.info(f"=== RENDER TAMAMLANDI! Dosya: {final_video_path} ({size_mb:.1f} MB | Süre: {total_time:.1f} sn) ===")
        return final_video_path


if __name__ == "__main__":
    plan_file = OUTPUT_DIR / "latest_plan.json"
    if not plan_file.exists():
        raise FileNotFoundError("latest_plan.json bulunamadı!")

    with open(plan_file, "r", encoding="utf-8") as f:
        plan_obj = ProductionPlanSchema.model_validate(json.load(f))

    renderer = VideoRenderer(plan_obj)
    renderer.render_full_video()
