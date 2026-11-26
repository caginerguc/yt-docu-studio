"""YouTube API kontrolü: python check_youtube.py  (GitHub Actions'ta 'YouTube API Kontrol' workflow'u çalıştırır)"""
import os
import sys

from src import config
from src.youtube_check import diagnose


def main() -> int:
    config.setup_logging()
    rep = diagnose()
    text = rep.text()
    print("\n=== YouTube API Kontrol Raporu ===\n" + text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:   # Actions çalıştırma sayfasında özet olarak görünür
        with open(summary, "a", encoding="utf-8") as f:
            f.write("## YouTube API Kontrol Raporu\n\n```\n" + text + "\n```\n")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
