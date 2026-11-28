"""Anlatıcı sesi denemesi: aynı belgesel paragrafını birkaç Almanca sesle seslendirir (GitHub Actions'ta
'Ses Denemesi' workflow'u çalıştırır). Bot gibi cümle cümle üretir ve cümleler arasına aynı duraklamayı koyar,
böylece videodakiyle aynı duyulur. Çıktı: output/voice_test/*.mp3 (+ LISTE.txt)

  python voice_test.py
"""
from __future__ import annotations

import asyncio
import base64
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

OUT = Path("output/voice_test")
SR = 48000
GAP_S = 0.35            # config.SENTENCE_GAP_S ile aynı
QUESTION_GAP_S = 0.9    # soru cümlesinden önce (pause_before: long)

TEXT = [
    "Im Jahr 1981 durchsucht der Steuerfahnder Klaus Förster Schließfächer einer Düsseldorfer Bank.",
    "In den beschlagnahmten Unterlagen findet er ein Kassenbuch des Flick-Generalbuchhalters.",
    "Die darin vermerkten Summen führen ohne Umwege direkt an die Spitze der damaligen Bundesrepublik.",
    "Was als Routineermittlung wegen Steuerhinterziehung begann, offenbart plötzlich ein systematisches "
    "Schattenreich verdeckter Zahlungen.",
    "Wie gelang es einem einzelnen Industrieimperium, über Jahre hinweg unbemerkt die politische Führung "
    "der Bundesrepublik finanziell einzubinden?",
]

# (dosya adı, ses, açıklama)
VOICES = [
    ("1_Killian", "de-DE-KillianNeural", "şu anki ses: net, genç, nötr"),
    ("2_Conrad", "de-DE-ConradNeural", "kalın, olgun, otoriter: klasik belgesel anlatıcısı"),
    ("3_Florian", "de-DE-FlorianMultilingualNeural", "yeni nesil erkek ses, daha doğal tonlama"),
    ("4_Seraphina", "de-DE-SeraphinaMultilingualNeural", "yeni nesil kadın ses, sıcak"),
]
# Google Cloud Text-to-Speech (Chirp 3 HD): yalnızca GOOGLE_TTS_API_KEY tanımlıysa denenir
GOOGLE_VOICES = [
    ("5_Google_Charon", "de-DE-Chirp3-HD-Charon", "Google Chirp 3 HD: bilgilendirici erkek ses"),
    ("6_Google_Algenib", "de-DE-Chirp3-HD-Algenib", "Google Chirp 3 HD: pürüzlü, kalın erkek ses"),
    ("7_Google_Orus", "de-DE-Chirp3-HD-Orus", "Google Chirp 3 HD: kararlı erkek ses"),
    ("8_Google_Sadaltager", "de-DE-Chirp3-HD-Sadaltager", "Google Chirp 3 HD: bilgili erkek ses"),
]
GOOGLE_RATES = [("a_094", 0.94, "%6 yavaş"), ("b_088", 0.88, "%12 yavaş, daha ağır")]
# Algenib, hız 0.94, ton düşürülmüş (yarım ton): botun ayarı -1.2
ALGENIB_PITCH = [("9_Algenib_ton-1.0", -1.0), ("9_Algenib_ton-1.2_BOT", -1.2), ("9_Algenib_ton-1.5", -1.5)]
# (ek, hız, ton, açıklama)
SETTINGS = [
    ("a_mevcut", "-8%", "+0Hz", "botun şu anki ayarı (%8 yavaş)"),
    ("b_derin", "-12%", "-4Hz", "daha ağır ve biraz daha kalın"),
]


async def _tts(text: str, voice: str, rate, pitch: str, path: Path) -> None:
    if "Chirp3" in voice:
        return _google(text, voice, rate, path)
    import edge_tts

    for attempt in range(3):
        try:
            await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(str(path))
            if path.exists() and path.stat().st_size > 2000:
                return
        except Exception as e:  # ağ / servis hatası: kısa bekleyip tekrar dene
            print(f"  uyarı: {voice} deneme {attempt + 1}: {e}", file=sys.stderr)
        await asyncio.sleep(2 + 3 * attempt)
    raise RuntimeError(f"{voice} seslendirilemedi")


def _google(text: str, voice: str, rate: float, path: Path) -> None:
    import requests

    body = {"input": {"text": text}, "voice": {"languageCode": voice.split("-Chirp3")[0], "name": voice},
            "audioConfig": {"audioEncoding": "MP3", "sampleRateHertz": 24000, "speakingRate": rate}}
    r = requests.post("https://texttospeech.googleapis.com/v1/text:synthesize",
                      params={"key": os.environ["GOOGLE_TTS_API_KEY"]}, json=body, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Google HTTP {r.status_code}: {r.text[:200]}")
    path.write_bytes(base64.b64decode(r.json()["audioContent"]))


def _decode_trimmed(mp3: Path) -> np.ndarray:
    """Botla aynı: baştaki/sondaki sessizliği kırp, 48 kHz mono."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mp3), "-af",
         "silenceremove=start_periods=1:start_threshold=-50dB,areverse,"
         "silenceremove=start_periods=1:start_threshold=-50dB,areverse",
         "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
        stdin=subprocess.DEVNULL, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16)


def _write_mp3(pcm: np.ndarray, path: Path) -> None:
    wav = path.with_suffix(".wav")
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.astype(np.int16).tobytes())
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                    "-ar", "44100", "-b:a", "160k", str(path)], stdin=subprocess.DEVNULL, check=True)
    wav.unlink()


def _pitch_shift(mp3: Path, semitones: float) -> None:
    f = 2 ** (semitones / 12)
    tmp = mp3.with_suffix(".p.mp3")
    for af in (f"rubberband=pitch={f:.5f}", f"asetrate=24000*{f:.5f},aresample=24000,atempo={1 / f:.5f}"):
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp3), "-af", af, "-ar", "24000", str(tmp)],
                           stdin=subprocess.DEVNULL, capture_output=True)
        if r.returncode == 0:
            tmp.replace(mp3)
            return
    raise RuntimeError("ton düşürülemedi")


async def make(name: str, voice: str, rate: str, pitch: str, out: Path, semitones: float = 0.0) -> float:
    tmp = OUT / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    parts: list[np.ndarray] = []
    for i, sentence in enumerate(TEXT):
        mp3 = tmp / f"{name}_{i}.mp3"
        await _tts(sentence, voice, rate, pitch, mp3)
        if semitones:
            _pitch_shift(mp3, semitones)
        if i:
            gap = QUESTION_GAP_S if sentence.rstrip().endswith("?") else GAP_S
            parts.append(np.zeros(int(gap * SR), np.int16))
        parts.append(_decode_trimmed(mp3))
    pcm = np.concatenate(parts)
    _write_mp3(pcm, out)
    return len(pcm) / SR


async def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["Ses denemesi: aynı paragraf, cümle cümle (bot gibi), aynı duraklamalarla.", ""]
    failed = 0
    for vname, voice, vdesc in VOICES:
        for sname, rate, pitch, sdesc in SETTINGS:
            out = OUT / f"{vname}_{sname}.mp3"
            try:
                secs = await make(f"{vname}_{sname}", voice, rate, pitch, out)
                lines.append(f"{out.name:28s} {voice}  hız {rate}, ton {pitch}  ({secs:.1f} sn)")
                print(f"tamam: {out.name} ({secs:.1f} sn)")
            except Exception as e:
                failed += 1
                lines.append(f"{out.name:28s} ÜRETİLEMEDİ: {e}")
                print(f"HATA: {out.name}: {e}", file=sys.stderr)
        lines.append(f"    -> {vdesc}")
        lines.append("")
    if os.environ.get("GOOGLE_TTS_API_KEY"):
        for vname, voice, vdesc in GOOGLE_VOICES:
            for sname, rate, sdesc in GOOGLE_RATES:
                out = OUT / f"{vname}_{sname}.mp3"
                try:
                    secs = await make(f"{vname}_{sname}", voice, rate, "", out)
                    lines.append(f"{out.name:34s} {voice}  hız {rate}  ({secs:.1f} sn)")
                    print(f"tamam: {out.name} ({secs:.1f} sn)")
                except Exception as e:
                    lines.append(f"{out.name:34s} ÜRETİLEMEDİ: {e}")
                    print(f"HATA: {out.name}: {e}", file=sys.stderr)
            lines.append(f"    -> {vdesc}")
            lines.append("")
        for name, semi in ALGENIB_PITCH:
            out = OUT / f"{name}.mp3"
            try:
                secs = await make(name, "de-DE-Chirp3-HD-Algenib", 0.94, "", out, semitones=semi)
                lines.append(f"{out.name:34s} Algenib  hız 0.94, ton {semi:+.1f} yarım ton  ({secs:.1f} sn)")
                print(f"tamam: {out.name} ({secs:.1f} sn)")
            except Exception as e:
                lines.append(f"{out.name:34s} ÜRETİLEMEDİ: {e}")
                print(f"HATA: {out.name}: {e}", file=sys.stderr)
        lines.append("    -> Algenib, biraz kalınlaştırılmış (_BOT = botun kullanacağı ayar)")
        lines.append("")
    else:
        lines += ["(Google sesleri denenmedi: GOOGLE_TTS_API_KEY tanımlı değil.)", ""]
    lines += ["Ayarlar:"] + [f"  {s[0]}: {s[3]}" for s in SETTINGS]
    (OUT / "LISTE.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for p in (OUT / "_tmp").glob("*"):
        p.unlink()
    (OUT / "_tmp").rmdir()
    total = len(VOICES) * len(SETTINGS)
    return 1 if failed == total else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
