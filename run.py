from dotenv import load_dotenv

from telegram_downloader.telegram_downloader import TelegramDownloader

load_dotenv()
import os

# print(os.getenv("TELETHON_SESSION_STR"))
print(os.getenv("TELEGRAM_API_ID"))
print(os.getenv("TELEGRAM_API_HASH"))


if __name__ == "__main__":
    TelegramDownloader(config_path="resources/config_debug.yaml").run()
