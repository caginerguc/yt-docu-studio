import os
import json
import logging
from pathlib import Path
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from src.config import OUTPUT_DIR
from src.agents_editorial import ProductionPlanSchema

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Stage6_YouTubeUploader")

YOUTUBE_TOKEN_URI = "https://oauth2.googleapis.com/token"
YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube"
]


class YouTubeUploader:
    def __init__(self):
        self.client_id = os.getenv("YT_CLIENT_ID")
        self.client_secret = os.getenv("YT_CLIENT_SECRET")
        self.refresh_token = os.getenv("YT_REFRESH_TOKEN")

    def is_configured(self) -> bool:
        """YouTube OAuth kimlik bilgilerinin tanımlı olup olmadığını kontrol eder."""
        return bool(self.client_id and self.client_secret and self.refresh_token)

    def _get_authenticated_service(self):
        """Tarayıcı gerektirmeden Refresh Token ile YouTube API v3 istemcisini oluşturur."""
        if not self.is_configured():
            raise EnvironmentError(
                "YouTube API kimlik bilgileri (YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN) eksik!"
            )

        creds = Credentials(
            token=None,
            refresh_token=self.refresh_token,
            token_uri=YOUTUBE_TOKEN_URI,
            client_id=self.client_id,
            client_secret=self.client_secret,
            scopes=YOUTUBE_SCOPES
        )
        # Access token'ı yenile
        creds.refresh(Request())
        return build("youtube", "v3", credentials=creds)

    def upload_documentary(
        self,
        video_path: Path,
        thumbnail_path: Path,
        plan: ProductionPlanSchema,
        privacy_status: str = "private"
    ) -> str:
        """Videoyu, SEO verilerini ve 16:9 kapak fotoğrafını YouTube'a yükler."""
        if not video_path.exists():
            raise FileNotFoundError(f"Yüklenecek video bulunamadı: {video_path}")

        youtube = self._get_authenticated_service()

        # Kategori 28: Science & Technology / 25: News & Politics
        body = {
            "snippet": {
                "title": plan.seo.youtube_title[:100],
                "description": plan.seo.youtube_description,
                "tags": plan.seo.tags[:15],
                "categoryId": "28",
                "defaultLanguage": plan.language.lower(),
                "defaultAudioLanguage": plan.language.lower()
            },
            "status": {
                "privacyStatus": privacy_status,  # "private", "unlisted" veya "public"
                "selfDeclaredMadeForKids": False,
                "containsSyntheticMedia": True    # YouTube AI/Sentetik medya şeffaflık beyanı
            }
        }

        logger.info(f"YouTube'a video yükleniyor: '{plan.seo.youtube_title}' (Gizlilik: {privacy_status})...")
        media = MediaFileUpload(
            str(video_path),
            mimetype="video/mp4",
            chunksize=10 * 1024 * 1024,  # 10 MB parçalar halinde kesintisiz yükleme
            resumable=True
        )

        request = youtube.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media
        )

        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                progress = int(status.progress() * 100)
                logger.info(f"Yükleme İlerlemesi: %{progress}")

        video_id = response["id"]
        logger.info(f"Video başarıyla yüklendi! YouTube Video ID: {video_id} (https://youtu.be/{video_id})")

        # Özel Thumbnail (Kapak Görseli) Yükleme (Hesabın telefon doğrulaması yapılmış olmalıdır)
        if thumbnail_path and thumbnail_path.exists():
            try:
                logger.info("16:9 Thumbnail kapak fotoğrafı yükleniyor...")
                youtube.thumbnails().set(
                    videoId=video_id,
                    media_body=MediaFileUpload(str(thumbnail_path), mimetype="image/jpeg")
                ).execute()
                logger.info("Thumbnail başarıyla ayarlandı!")
            except Exception as e:
                logger.warning(f"Thumbnail yüklenemedi (Kanal telefon doğrulaması eksik olabilir): {e}")

        return video_id


if __name__ == "__main__":
    uploader = YouTubeUploader()
    print("YouTube Uploader Modülü Hazır! Kimlik Bilgileri Tanımlı mı?:", uploader.is_configured())
