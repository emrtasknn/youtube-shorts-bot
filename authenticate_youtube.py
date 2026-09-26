"""
One-time YouTube OAuth authenticator.
Reads client_secrets.json and generates token.json for local runs and GitHub Actions CI.
"""
import sys
import webbrowser
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
CLIENT_SECRETS_FILE = Path("client_secrets.json")
TOKEN_FILE = Path("token.json")


def authenticate():
    if not CLIENT_SECRETS_FILE.exists():
        print(f"HATA: '{CLIENT_SECRETS_FILE}' bulunamadı.", flush=True)
        print("Lütfen Google Cloud Console'dan indirdiğiniz OAuth dosyasını proje köküne 'client_secrets.json' adıyla koyun.", flush=True)
        sys.exit(1)

    print("YouTube OAuth kimlik doğrulama akışı hazırlanıyor...", flush=True)
    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS_FILE), SCOPES)

    prompt_msg = (
        "\n" + "=" * 70 + "\n"
        "Tarayıcınız otomatik açılmazsa aşağıdaki bağlantıyı kopyalayıp tarayıcınızda açın:\n\n"
        "{url}\n" +
        "=" * 70 + "\n"
    )

    creds = flow.run_local_server(
        port=0,
        authorization_prompt_message=prompt_msg,
        open_browser=True,
        access_type="offline",
        prompt="consent",
    )

    TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
    print(f"\n✓ BAŞARILI: '{TOKEN_FILE}' başarıyla üretildi!", flush=True)
    if creds.refresh_token:
        print("\n" + "=" * 70, flush=True)
        print("YOUTUBE_REFRESH_TOKEN (BUNU SADECE RENDER ENVIRONMENT'A EKLE):", flush=True)
        print(creds.refresh_token, flush=True)
        print("=" * 70 + "\n", flush=True)
    else:
        print("\nUYARI: Refresh token dönmedi. OAuth akışını tekrar çalıştırmak gerekebilir.\n", flush=True)
    return creds


if __name__ == "__main__":
    authenticate()
