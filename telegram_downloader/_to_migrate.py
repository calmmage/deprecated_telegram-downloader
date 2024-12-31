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


async def get_chats():
    logger.debug("Getting chats asynchronously")
    if _has_fresh_chats_on_disk():
        logger.debug("Found fresh chats on disk")
        return _get_chats_from_disk()
    else:
        logger.debug("No fresh chats found, getting from client")
        chats = await _get_chats_from_client()
        _save_chats_to_disk(chats)
        return chats


# Create constructor map
type_map = {
    "Channel": lambda d: Channel(**d),
    "User": lambda d: User(**d),
    "Chat": lambda d: Chat(**d),
}


def _load_chats_from_json(filepath):
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    chats = []
    for item in data["chats"]:

        # Parse the JSON string back to dict
        entity_data = json.loads(item["entity"])
        entity_type = entity_data.pop("_")
        if entity_type == "ChatForbidden":
            logger.warning(f"Skipping 'forbidden chat': {entity_data}")
            continue

        # Construct the entity
        entity = type_map[entity_type](entity_data)
        chat = {
            "entity": entity,
            "last_message_date": (
                datetime.fromisoformat(item["last_message_date"])
                if item["last_message_date"]
                else None
            ),
        }
        chats.append(chat)

    logger.info(f"Loaded {len(chats)} entities from {filepath}")
    return chats


def _get_chats_from_disk():
    logger.debug("Loading chats from disk")
    data_path = Path("data/chats.json")

    chats = _load_chats_from_json(data_path)

    logger.debug(f"Loaded {len(chats)} chats from disk")
    return chats


async def _get_chat_list(client=None) -> List[Dialog]:
    logger.debug("Getting chat list from Telegram")
    if client is None:
        client = await get_telethon_client()
    chats = await client.get_dialogs()
    logger.debug(f"Retrieved {len(chats)} dialogs from Telegram")
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


async def _get_chats_from_client():
    logger.debug("Getting chats from client asynchronously")
    # Store chats in memory
    chats = await _get_chat_list()
    logger.debug(f"Retrieved {len(chats)} chats")
    result = []
    for chat in chats:
        if type(chat.entity).__name__ == "ChatForbidden":
            logger.warning(f"Skipping 'forbidden chat': {chat.entity}")
            continue
        result.append({"entity": chat.entity, "last_message_date": chat.date})
    return result


def _has_fresh_chats_on_disk(max_age: timedelta = timedelta(days=1)):
    logger.debug("Checking for fresh chats on disk")
    data_path = Path("data/chats.json")

    if not data_path.exists():
        logger.debug("No chats file found on disk")
        return False

    data = json.loads(data_path.read_text())
    timestamp = data["timestamp"]
    is_fresh = datetime.now() - datetime.fromisoformat(timestamp) < max_age
    logger.debug(f"Chats on disk are {'fresh' if is_fresh else 'stale'}")
    return is_fresh


def _save_chats_to_disk(chats):
    logger.debug(f"Saving {len(chats)} chats to disk")
    data = {
        "chats": [
            {
                "entity": chat["entity"].to_json(),
                "last_message_date": (
                    chat["last_message_date"].isoformat() if chat["last_message_date"] else None
                ),
            }
            for chat in chats
        ],
        "timestamp": datetime.now().isoformat(),
    }

    data_path = Path("data/chats.json")
    data_path.parent.mkdir(exist_ok=True)
    data_path.write_text(json.dumps(data, indent=2))
    logger.debug("Chats saved successfully")


def _save_chats_to_db(chats, db, collection_name: str = "telegram_chats"):
    # Drop old items from collection
    db[collection_name].delete_many({})
    logger.debug(f"Dropped old items from {collection_name}")

    # Insert new items
    timestamp = datetime.now().isoformat()
    data = [
        {
            "timestamp": timestamp,
            "data": chat["entity"].to_json(),
            "last_message_date": (
                chat["last_message_date"].isoformat() if chat["last_message_date"] else None
            ),
        }
        for chat in chats
    ]

    db[collection_name].insert_many(data)
    logger.debug(f"Saved {len(data)} chats to db")


def _load_chats_from_db(db, collection_name: str = "telegram_chats"):
    collection = db[collection_name]
    data = collection.find()

    chats = []
    for item in data:

        # Parse the JSON string back to dict
        entity_data = json.loads(item["data"])
        entity_type = entity_data.pop("_")
        if entity_type == "ChatForbidden":
            logger.warning(f"Skipping 'forbidden chat': {entity_data}")
            continue

        # Construct the entity
        entity = type_map[entity_type](entity_data)
        chat = {
            "entity": entity,
            "last_message_date": (
                datetime.fromisoformat(item["last_message_date"])
                if item["last_message_date"]
                else None
            ),
        }
        chats.append(chat)

    logger.info(f"Loaded {len(chats)} entities from db collection {collection_name}")
    return chats


def _has_fresh_chats_on_db(
    db, collection_name: str = "chats", max_age: timedelta = timedelta(days=1)
):
    collection = db[collection_name]
    chats = collection.find()
    logger.debug(f"Loaded {len(chats)} chats from db")
    if len(chats) == 0:
        return False
    timestamp = chats[0]["timestamp"]
    is_fresh = datetime.now() - datetime.fromisoformat(timestamp) < max_age
    logger.debug(f"Chats on db are {'fresh' if is_fresh else 'stale'}")
    return is_fresh


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


def calculate_stats(
    chats, big_threshold: int = 1000, recent_threshold: timedelta = timedelta(days=30)
):
    """
    Calculate stats about chats

    Basic:
    - private messages
    - group chats
    - channels
    - bots

    Advanced:
    - owned groups
    - owned bots
    - owned channels

    Big:
    - big groups (> 1000)
    - big channels
    """
    logger.debug("Calculating chat statistics")
    stats = defaultdict(int)
    # Basic:
    for chat in chats:
        entity = chat["entity"]
        last_message_date = chat["last_message_date"]

        category = categorize_entity(entity)
        stats[category + "s"] += 1

        # last_message_date = datetime.fromisoformat(getattr(chat, 'date', None)) if getattr(chat, 'date', None) else None
        recent = (
            last_message_date
            and datetime.now(last_message_date.tzinfo) - last_message_date < recent_threshold
        )
        if recent:
            stats[f"recent_{category}s"] += 1

        owner = getattr(entity, "creator", False)
        if owner:
            stats[f"owned_{category}s"] += 1
            if recent:
                stats[f"recent_owned_{category}s"] += 1

        members = getattr(entity, "participants_count", 0)
        if members > big_threshold:
            stats[f"big_{category}s"] += 1

    # Bonus: size / time-based stats
    # for each of the above - count which have recent messages (last 30 days)
    logger.debug(f"Calculated stats: {stats}")
    return stats


def is_recent(chat, recent_threshold: timedelta = timedelta(days=30)):
    last_message_date = chat["last_message_date"]
    return (
        last_message_date
        and datetime.now(last_message_date.tzinfo) - last_message_date < recent_threshold
    )


def get_random_chat(
    chats,
    entity_type: str = None,  # 'group', 'channel', 'bot', 'private chat'
    owned: bool = None,
    recent: bool = None,
    big: bool = None,
    recent_threshold: timedelta = timedelta(days=30),
    big_threshold: int = 1000,
):
    if entity_type is not None:
        chats = [chat for chat in chats if categorize_entity(chat["entity"]) == entity_type]
    if len(chats) == 0:
        logger.warning(f"No chats found for {entity_type=}")
        return None
    if owned is not None:
        is_owned = lambda chat: getattr(chat["entity"], "creator", False)
        chats = [chat for chat in chats if is_owned(chat) == owned]
        if len(chats) == 0:
            logger.warning(f"No chats found for {entity_type=} {owned=}")
            return None
    if recent is not None:
        chats = [chat for chat in chats if is_recent(chat, recent_threshold) == recent]
        if len(chats) == 0:
            logger.warning(f"No chats found for {entity_type=} {recent=}")
            return None
    if big is not None:
        is_big = lambda chat: getattr(chat["entity"], "participants_count", 0) > big_threshold
        chats = [chat for chat in chats if is_big(chat) == big]
    if len(chats) == 0:
        logger.warning(f"No chats found for {entity_type=} {owned=} {recent=} {big=}")
        return None
    return random.choice(chats)


# endregion calculate_stats


# region plot_stats
def plot_size_distribution(items, title, output_path: Path):

    import matplotlib.pyplot as plt
    import numpy as np

    logger.debug(f"Plotting size distribution for {title}")
    sizes = [getattr(item.entity, "participants_count", 0) for item in items]
    max_size = max(sizes)

    # Custom bin edges for more intuitive intervals
    bins = [
        1,
        3,
        10,
        30,
        100,
        300,  # Small groups/channels
        1000,
        3000,
        10000,  # Medium
        30000,
        100000,  # Large
        300000,
        1000000,  # Huge
    ]

    # Filter out empty upper bins
    while bins[-1] > max_size * 1.1:  # Keep one empty bin for visual clarity
        bins.pop()

    plt.figure(figsize=(12, 6))

    # Calculate histogram data
    counts, edges = np.histogram(sizes, bins=bins)

    # Plot bars manually with fixed width in log space
    bar_width = 0.8  # Width in log space
    log_edges = np.log10(edges[:-1])

    plt.bar(
        edges[:-1],
        counts,
        width=[edge * bar_width for edge in edges[:-1]],  # Width proportional to x position
        align="center",
    )

    plt.xscale("log")

    # Create interval labels
    labels = [f"{bins[i]:,}-{bins[i+1]:,}" for i in range(len(bins) - 1)]
    plt.xticks(bins[:-1], labels, rotation=45, ha="right")

    # Add value labels on top of bars
    for i, count in enumerate(counts):
        if count > 0:  # Only label non-empty bars
            plt.text(edges[i], count, f"{int(count)}", va="bottom", ha="center")

    plt.title(f"{title} Size Distribution (n={len(items)})")
    plt.xlabel("Number of participants")
    plt.ylabel("Count")

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save plot to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    # Print statistics
    print(f"\n{title} size statistics:")
    print(f"Median size: {np.median(sizes):,.0f}")
    print(f"Mean size: {np.mean(sizes):,.0f}")
    print(f"Max size: {max_size:,}")
    logger.debug(f"Finished plotting size distribution for {title}")


# # Example usage:
# plot_path = Path("data/plots/groups_distribution.png")
# plot_size_distribution(groups, 'Groups', plot_path)
# # To display inline in notebook:
# # from IPython.display import Image
# # Image(filename=str(plot_path))


def plot_small_size_distribution(items, title, output_path: Path):

    import matplotlib.pyplot as plt
    import numpy as np

    logger.debug(f"Plotting small size distribution for {title}")
    sizes = [getattr(item.entity, "participants_count", 0) for item in items]

    # Custom bin edges for small groups
    bins = list(range(1, 12))  # 1 to 11 to capture sizes 1-10

    plt.figure(figsize=(10, 6))

    # Calculate histogram data
    counts, edges = np.histogram(sizes, bins=bins)

    # Plot bars
    plt.bar(range(1, 11), counts, width=0.7, align="center")

    # Set x-axis ticks to whole numbers
    plt.xticks(range(1, 11))

    # Add value labels on top of bars
    for i, count in enumerate(counts):
        if count > 0:  # Only label non-empty bars
            plt.text(i + 1, count, f"{int(count)}", va="bottom", ha="center")

    plt.title(f"{title} Size Distribution (1-10 members, n={sum(counts)})")
    plt.xlabel("Number of participants")
    plt.ylabel("Count")

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save plot to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    logger.debug(f"Finished plotting small size distribution for {title}")


# # Example usage:
# plot_path = Path("data/plots/small_groups_distribution.png")
# plot_small_size_distribution(groups, 'Groups', plot_path)
# # To display inline in notebook:
# # from IPython.display import Image
# # Image(filename=str(plot_path))

# endregion plot_stats


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


async def main(debug: bool = False):
    setup_logger(logger, level="DEBUG" if debug else "INFO")
    logger.debug("Starting main function")
    chats = await get_chats()
    logger.debug("Main function completed")
    logger.debug(f"Loaded {len(chats)} chats")

    stats = calculate_stats(chats)
    logger.info("Chat statistics:")
    for key, value in stats.items():
        logger.info(f"{key}: {value}")

    random_chat = get_random_chat(chats, "group", recent=True, big=False)
    logger.info(f"Random recent group: {random_chat['entity'].title}")

    random_chat = get_random_chat(chats, "group", recent=False, big=False)
    logger.info(f"Random old group: {random_chat['entity'].title}")

    random_chat = get_random_chat(chats, "channel", big=False)
    logger.info(f"Random small channel: {random_chat['entity'].title}")

    random_chat = get_random_chat(chats, "channel", big=True)
    logger.info(f"Random big channel: {random_chat['entity'].title}")

    random_chat = get_random_chat(chats, "bot")
    logger.info(f"Random bot: {random_chat['entity'].username}")

    random_chat = get_random_chat(chats, "private chat", recent=True)
    logger.info(f"Random private chat: {random_chat['entity'].username}")

    random_chat = get_random_chat(chats, "private chat", recent=False)
    logger.info(f"Random old private chat: {random_chat['entity'].username}")


# endregion main
# endregion 1
# region 2
# from draft_3.main import get_telethon_client, get_database
"""
# - 1 - database
# - 2 - connection to telethon, session
# - 3 - get chats
# - 4 - yaml config
# - 5 - get messages
"""


# endregion 1


# region 2 - connection to telethon, session
async def get_telethon_client() -> TelegramClient:

    # Example initialization:
    SESSIONS_DIR = Path("sessions")
    SESSIONS_DIR.mkdir(exist_ok=True)
    TELEGRAM_API_ID = os.getenv("TELEGRAM_API_ID")
    if TELEGRAM_API_ID is None:
        raise ValueError("TELEGRAM_API_ID is not set")
    TELEGRAM_API_ID = int(TELEGRAM_API_ID)
    TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH")
    if TELEGRAM_API_HASH is None:
        raise ValueError("TELEGRAM_API_HASH is not set")
    telethon_manager = TelethonClientManager(
        storage_mode=StorageMode.TO_DISK,
        api_id=TELEGRAM_API_ID,
        api_hash=TELEGRAM_API_HASH,
        sessions_dir=SESSIONS_DIR,
    )

    user_id = os.getenv("TELEGRAM_USER_ID")

    # Get client for user
    client = await telethon_manager.get_telethon_client(int(user_id))

    if not client:
        raise ValueError("Failed to get client")

    return client


# region 4
# from draft_3.model_message_donwload_config import MessageDownloadConfig, ChatCategoryConfig


# endregion 4
