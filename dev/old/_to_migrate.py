import asyncio
import json
import os
import random
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from loguru import logger
from pymongo import MongoClient  # todo: change to motor
from telethon import TelegramClient
from telethon.tl.custom import Dialog
from telethon.tl.types import Channel, Chat, User

from telegram_downloader.telethon_client_manager import StorageMode, TelethonClientManager
from telegram_downloader.utils import setup_logger


# region 1
# from draft_3.get_chat_list import get_chats
# idea 1: just load chat list
# idea 2: cache chat list. Save timestamp. If timestamp is older than 1 day, load chat list.
# idea 3: add --refresh flag. If flag is set, load chat list from scratch.
# idea 4: calculate fancy stats about chats.
# Basic:
# private messages
# group chats
# channels
# bots

# Advanced:
# owned groups
# owned bots
# owned channels

# big groups (> 1000)
# big channels

# Bonus: size / time-based stats
# for each of the above - count which have recent messages (last 30 days)


# region get_chats
def get_chats_sync():
    logger.debug("Getting chats synchronously")
    if _has_fresh_chats_on_disk():
        logger.debug("Found fresh chats on disk")
        return _get_chats_from_disk()
    else:
        logger.debug("No fresh chats found, getting from client")
        chats = _get_chats_from_client_sync()
        _save_chats_to_disk(chats)
        return chats


def _get_chats_from_client_sync():
    logger.debug("Getting chats from client synchronously")
    # Store chats in memory
    chats = asyncio.run(_get_chat_list())
    logger.debug(f"Retrieved {len(chats)} chats")
    chats = []
    for chat in chats:
        if type(chat.entity).__name__ == "ChatForbidden":
            logger.warning(f"Skipping 'forbidden chat': {chat.entity}")
            continue
        chats.append(chat.entity)
    return chats


# endregion get_chats

# region calculate_stats


def categorize_entity(entity):
    if isinstance(entity, User):
        return "bot" if getattr(entity, "bot", False) else "private chat"
    elif isinstance(entity, Chat):
        return "group"
    elif isinstance(entity, Channel):
        return "channel" if not entity.megagroup else "group"
    return "unknown"


def is_recent(chat, recent_threshold: timedelta = timedelta(days=30)):
    last_message_date = chat["last_message_date"]
    return (
        last_message_date
        and datetime.now(last_message_date.tzinfo) - last_message_date < recent_threshold
    )


# region main
async def main_linear(debug: bool = False):
    setup_logger(logger, level="DEBUG" if debug else "INFO")

    logger.debug("step 0: check chats on disk. should result false")
    chats_on_disk = _has_fresh_chats_on_disk()
    logger.debug(f"chats_on_disk: {chats_on_disk}")

    logger.debug("step 1: load chats from client")
    chats = await _get_chats_from_client()
    logger.debug(f"Retrieved {len(chats)} chats from client")

    logger.debug("step 2: save chats to disk")
    _save_chats_to_disk(chats)
    chats_on_disk = _has_fresh_chats_on_disk()
    logger.debug(f"chats_on_disk: {chats_on_disk}")

    logger.debug("step 3: load chats from disk")
    chats = _get_chats_from_disk()
    logger.debug(f"Retrieved {len(chats)} chats from disk")


async def main_db(debug: bool = False):
    setup_logger(logger, level="DEBUG" if debug else "INFO")
    logger.debug("Starting main function")
    chats = await get_chats()
    logger.debug(f"Loaded {len(chats)} chats from disk")

    db = get_database()
    _save_chats_to_db(chats, db)
    logger.debug("Chats saved to db")

    chats = _load_chats_from_db(db)
    logger.debug("Chats loaded from db")
    logger.debug(f"Loaded {len(chats)} chats")


# endregion main
