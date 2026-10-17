import json
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube"
]

def generate_refresh_token():
    secret_files = list(Path(".").glob("client_secret*.json"))
    if not secret_files:
        print("❌ HATA: Ana dizinde 'client_secret.json' bulunamadı!")
        print("1. https://console.cloud.google.com/ adresine gidin.")
        print("2. 'YouTube Data API v3' kütüphanesini etkinleştirin.")
        print("3. OAuth 2.0 İstemci Kimliği (Masaüstü Uygulaması / Desktop App) oluşturup JSON dosyasını bu klasöre indirin.")
        return

    secret_path = secret_files[0]
    print(f"🔑 Bulunan kimlik dosyası: {secret_path.name}")
    flow = InstalledAppFlow.from_client_secrets_file(str(secret_path), SCOPES)
    creds = flow.run_local_server(port=8080, prompt="consent", access_type="offline")

    raw_data = json.loads(secret_path.read_text(encoding="utf-8"))
    installed = raw_data.get("installed") or raw_data.get("web") or {}

    print("\n" + "="*60)
    print("✅ YOUTUBE OAUTH 2.0 YETKİLENDİRMESİ BAŞARILI!")
    print("Aşağıdaki 3 değeri .env dosyanıza ve GitHub Actions Secrets bölümüne ekleyin:")
    print("="*60)
    print(f'YT_CLIENT_ID="{installed.get("client_id")}"')
    print(f'YT_CLIENT_SECRET="{installed.get("client_secret")}"')
    print(f'YT_REFRESH_TOKEN="{creds.refresh_token}"')
    print("="*60)

if __name__ == "__main__":
    generate_refresh_token()
