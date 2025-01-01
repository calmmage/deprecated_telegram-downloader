import asyncio

from dotenv import load_dotenv
from loguru import logger

from telegram_downloader.telegram_downloader import TelegramDownloader
from telegram_downloader.utils import setup_logger


def main():
    load_dotenv()
    import argparse

    parser = argparse.ArgumentParser(description="Telegram Downloader")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--config", type=str, default="resources/config_archive.yaml", help="Path to config file"
    )
    parser.add_argument("--chat-sample", type=int, help="Number of random chats to process")
    parser.add_argument(
        "--ignore-finished", action="store_true", help="Ignore finished_downloading flag"
    )
    args = parser.parse_args()

    logger.debug("Setting up logger...")
    setup_logger(logger, level="DEBUG" if args.debug else "INFO")

    downloader = TelegramDownloader(config_path=args.config)

    asyncio.run(
        downloader.main(
            # debug=args.debug
            ignore_finished=args.ignore_finished,
            chat_sample_size=args.chat_sample,
        )
    )


if __name__ == "__main__":
    main()
