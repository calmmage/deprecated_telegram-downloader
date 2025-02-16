"""

"""

import asyncio
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from dotenv import load_dotenv
from loguru import logger

from telegram_downloader.telegram_downloader import TelegramDownloader
from telegram_downloader.utils import setup_logger


class TelegramAutoDownloader:
    def __init__(self, config_path: str = "resources/config_archive.yaml", interval_hours: int = 6):
        load_dotenv()
        setup_logger(logger)
        self.downloader = TelegramDownloader(config_path=config_path)
        self.scheduler = BlockingScheduler()
        self.interval_hours = interval_hours

    async def download_messages(self):
        logger.info(f"Starting scheduled download at {datetime.now()}")
        await self.downloader.main(ignore_finished=True)
        logger.info("Scheduled download completed")

    def run_download(self):
        asyncio.run(self.download_messages())

    def start(self):
        logger.info(f"Starting auto-downloader with {self.interval_hours} hour interval")
        self.scheduler.add_job(
            self.run_download,
            trigger=IntervalTrigger(hours=self.interval_hours),
            next_run_time=datetime.now(),  # Run immediately on start
        )
        self.scheduler.start()


if __name__ == "__main__":
    auto_downloader = TelegramAutoDownloader()
    auto_downloader.start()
