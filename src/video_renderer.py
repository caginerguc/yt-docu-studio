import json
import logging
import re
import subprocess
from pathlib import Path
from typing import List

from src.config import OUTPUT_DIR, MUSIC_DIR
from src.agents_editorial import ProductionPlanSchema

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Stage4_CinematicRenderer")


class VideoRenderer:
    def __init__(self, plan: ProductionPlanSchema):
        self.plan = plan
        self.scenes_dir = OUTPUT_DIR / "scenes"
        self.audio_dir = OUTPUT_DIR / "audio"
        self.clips_dir = OUTPUT_DIR / "clips"
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.final_video_path = OUTPUT_DIR / "final_documentary.mp4"
        self.master_srt_path = OUTPUT_DIR / "master_subtitles.srt"

    def get_audio_duration(self, audio_path: Path) -> float:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(res.stdout.strip())

    def ensure_valid_image(self, image_path: Path):
        if not image_path.exists() or image_path.stat().st_size < 5000:
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=#0c1017:s=1920x1080:d=1",
                "-frames:v", "1", str(image_path)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    def _build_cinematic_filter(self, effect: str, duration: float, fps: int = 30) -> str:
        frames = max(int(duration * fps), 30)
        if effect == "zoom_in":
            z_expr = f"min(zoom+{0.14/frames:.6f},1.14)"
            x_expr, y_expr = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
        elif effect == "zoom_out":
            z_expr = f"if(eq(on,1),1.14,max(1.0,zoom-{0.14/frames:.6f}))"
            x_expr, y_expr = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
        elif effect == "pan_left_to_right":
            z_expr = "1.12"
            x_expr = f"(iw-iw/zoom)*(on/{frames})"
            y_expr = "ih/2-(ih/zoom/2)"
        else:
            z_expr = "1.12"
            x_expr = f"(iw-iw/zoom)*(1-on/{frames})"
            y_expr = "ih/2-(ih/zoom/2)"

        # 4K Supersample Zoompan + Sinematik Noir Kontrast + 35mm Film Grain + Vignette (Köşe Karartma) + Fade
        fade_out_start = max(0.1, duration - 0.35)
        return (
            f"scale=3840:2160:force_original_aspect_ratio=increase,crop=3840:2160,"
            f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':d={frames}:s=1920x1080:fps={fps},"
            f"eq=contrast=1.08:brightness=-0.03:saturation=0.88,"
            f"noise=alls=6:allf=t+u,"
            f"vignette=PI/4,"
            f"fade=t=in:st=0:d=0.35,fade=t=out:st={fade_out_start:.2f}:d=0.35,"
            f"format=yuv420p"
        )

    def render_scene_clip(self, scene_num: int, effects: List[str]) -> float:
        audio_path = self.audio_dir / f"scene_{scene_num}.mp3"
        output_clip = self.clips_dir / f"clip_{scene_num}.mp4"

        total_duration = self.get_audio_duration(audio_path) + 0.40
        shot_images = sorted(self.scenes_dir.glob(f"scene_{scene_num}_*.jpg"))
        if not shot_images:
            single = self.scenes_dir / f"scene_{scene_num}.jpg"
            self.ensure_valid_image(single)
            shot_images = [single]

        num_shots = len(shot_images)
        shot_duration = total_duration / num_shots
        sub_clips = []

        logger.info(f"Sahne {scene_num} ({num_shots} kamera açısı, {total_duration:.1f}s) 35mm Noir filtreyle işleniyor...")

        for i, img_path in enumerate(shot_images):
            self.ensure_valid_image(img_path)
            sub_clip = self.clips_dir / f"clip_{scene_num}_shot_{i+1}.mp4"
            eff = effects[i % len(effects)] if effects else "zoom_in"
            vf = self._build_cinematic_filter(eff, shot_duration)

            cmd_shot = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", str(img_path),
                "-vf", vf,
                "-t", f"{shot_duration:.2f}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-an", str(sub_clip)
            ]
            subprocess.run(cmd_shot, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
            sub_clips.append(sub_clip)

        # Alt çekimleri (shot) tek bir sahne videosunda birleştir ve sesi + geçiş sub-bass vuruşunu ekle
        shot_list_file = self.clips_dir / f"shots_{scene_num}.txt"
        with open(shot_list_file, "w", encoding="utf-8") as f:
            for sc in sub_clips:
                f.write(f"file '{sc.resolve().as_posix()}'\n")

        cmd_merge = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(shot_list_file),
            "-i", str(audio_path),
            "-filter_complex", "[1:a]apad=pad_dur=0.40[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-t", f"{total_duration:.2f}",
            str(output_clip)
        ]
        subprocess.run(cmd_merge, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        return total_duration

    def _shift_srt_timestamps(self, srt_content: str, offset_sec: float) -> str:
        def parse_time(t_str: str) -> float:
            h, m, s_ms = t_str.split(":")
            s, ms = s_ms.split(",")
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

        def format_time(sec: float) -> str:
            h = int(sec // 3600)
            m = int((sec % 3600) // 60)
            s = int(sec % 60)
            ms = int(round((sec - int(sec)) * 1000))
            if ms >= 1000:
                s += 1
                ms -= 1000
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

        pattern = re.compile(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})")
        return pattern.sub(lambda m: f"{format_time(parse_time(m.group(1)) + offset_sec)} --> {format_time(parse_time(m.group(2)) + offset_sec)}", srt_content)

    def build_master_srt(self, durations: List[float]):
        master_blocks = []
        current_offset = 0.0
        counter = 1

        for idx, scene in enumerate(self.plan.scenes):
            srt_file = self.audio_dir / f"scene_{scene.scene_number}.srt"
            if srt_file.exists():
                raw = srt_file.read_text(encoding="utf-8").strip()
                shifted = self._shift_srt_timestamps(raw, current_offset)
                for block in re.split(r"\n\s*\n", shifted):
                    lines = block.strip().splitlines()
                    if len(lines) >= 2:
                        time_line = lines[1] if "-->" in lines[1] else lines[0]
                        text_lines = lines[2:] if "-->" in lines[1] else lines[1:]
                        master_blocks.append(f"{counter}\n{time_line}\n" + "\n".join(text_lines))
                        counter += 1
            current_offset += durations[idx]

        self.master_srt_path.write_text("\n\n".join(master_blocks) + "\n", encoding="utf-8")

    def ensure_background_music(self, total_duration: float) -> Path:
        chosen = self.plan.supervision.background_music
        if not chosen or chosen == "none":
            chosen = "cinematic_tension.mp3"

        real_mp3 = MUSIC_DIR / chosen
        if real_mp3.exists() and real_mp3.stat().st_size > 20000:
            return real_mp3

        synth_mp3 = OUTPUT_DIR / "dark_cinema_score.mp3"
        logger.info("Dark Cello & Oppenheimer tarzı Gerilim Ambiyansı (Sub-Bass Drone) sentezleniyor...")

        # Re Minör (D Minor) Karanlık Cello & Sub-Bass gerilim harmonikleri + Sahne başı derin nabız vuruşu
        expr = (
            "0.25*sin(2*PI*73.42*t*(1+0.002*sin(2*PI*0.15*t)))+"
            "0.18*sin(2*PI*110.00*t)+"
            "0.14*sin(2*PI*146.83*t)+"
            "0.10*sin(2*PI*174.61*t)*(0.5+0.5*sin(2*PI*0.08*t))+"
            "0.22*sin(2*PI*36.71*t)*exp(-mod(t,6.0)*1.8)"
        )

        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"aevalsrc='{expr}':s=44100:d={int(total_duration) + 15}",
            "-af", "lowpass=f=650,tremolo=f=0.20:d=0.45,volume=0.24",
            "-c:a", "libmp3lame", "-b:a", "192k",
            str(synth_mp3)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        return synth_mp3

    def assemble_final_video(self, total_duration: float):
        concat_list = self.clips_dir / "concat_list.txt"
        with open(concat_list, "w", encoding="utf-8") as f:
            for scene in self.plan.scenes:
                clip_path = (self.clips_dir / f"clip_{scene.scene_number}.mp4").resolve()
                f.write(f"file '{clip_path.as_posix()}'\n")

        merged_no_sub = OUTPUT_DIR / "merged_raw.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list), "-c", "copy", str(merged_no_sub)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

        bg_music = self.ensure_background_music(total_duration)
        srt_escaped = self.master_srt_path.resolve().as_posix().replace(":", r"\:")
        sub_filter = (
            f"subtitles='{srt_escaped}':force_style='FontName=DejaVu Sans,FontSize=15,"
            f"PrimaryColour=&H0000E6FF,OutlineColour=&H00000000,BackColour=&H99000000,"
            f"Bold=1,BorderStyle=4,Outline=1,Shadow=0,MarginV=30,Alignment=2'"
        )

        filter_complex = (
            f"[1:a]aloop=loop=-1:size=2e+09,atrim=0:{total_duration:.2f},volume=0.14[bg];"
            f"[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout];"
            f"[0:v]{sub_filter}[vout]"
        )

        cmd_final = [
            "ffmpeg", "-y",
            "-i", str(merged_no_sub),
            "-i", str(bg_music),
            "-filter_complex", filter_complex,
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            str(self.final_video_path)
        ]
        subprocess.run(cmd_final, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        logger.info(f"=== V3.0 Sinematik Belgesel Hazır: {self.final_video_path} ===")

    def run_pipeline(self) -> Path:
        # Eski klipleri temizle
        for old_clip in self.clips_dir.glob("*.mp4"):
            old_clip.unlink(missing_ok=True)

        durations = []
        for scene in self.plan.scenes:
            effects = getattr(scene, "camera_effects", None) or ["zoom_in", "pan_left_to_right", "zoom_out"]
            dur = self.render_scene_clip(scene.scene_number, effects)
            durations.append(dur)

        self.build_master_srt(durations)
        self.assemble_final_video(sum(durations))
        return self.final_video_path
