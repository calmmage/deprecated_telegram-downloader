import asyncio
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

from _to_migrate import get_chats
from dotenv import load_dotenv
from loguru import logger
from pymongo import MongoClient
from telethon import TelegramClient
from telethon.types import Message
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from telegram_downloader.config import (
    ChatCategoryConfig,
    TelegramDownloaderConfig,
    TelegramDownloaderEnvSettings,
)
from telegram_downloader.data_model import ChatData
from telegram_downloader.telethon_client_manager import StorageMode, TelethonClientManager
from telegram_downloader.utils import setup_logger


class TelegramDownloader:
    """
    # Plan

    # 1. load list of chats
    # 2. load config
    # 3. for each chat - pick a config that applies to it
    # 4. download messages as per config
    # 5. save messages to a database
    """

    def __init__(self, config_path: Path | str = Path("config.yaml"), **kwargs):
        # Load environment variables
        self.env = TelegramDownloaderEnvSettings(**kwargs)

        # Load YAML config
        config_path = Path(config_path)
        self.config = TelegramDownloaderConfig.from_yaml(config_path, **kwargs)

        self._db = None
        self._telethon_client = None

    def run(self):

        load_dotenv()

        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        args = parser.parse_args()

        # Run the async main function
        logger.debug("Starting script")
        asyncio.run(self.main(debug=args.debug))
        logger.debug("Script completed")

    async def main(self, debug: bool = False):
        """
        # 1. load list of chats
        # 2. load config
        # 3. for each chat - pick a config that applies to it
        # 4. download messages as per config
        # 5. save messages to a database
        """
        logger.debug("Setting up logger...")
        setup_logger(logger, level="DEBUG" if debug else "INFO")
        logger.debug("Starting script")

        # 1. load list of chats
        logger.debug("Loading chats...")
        chats = await self._load_chats()
        logger.debug(f"Loaded {len(chats)} chats")

        if debug:
            logger.debug("Debug mode enabled - shuffling and limiting chats")
            random.shuffle(chats)
            chats = chats[:10]
            logger.debug(f"Limited to {len(chats)} chats")

        # 2. load config
        logger.debug("Loading config...")
        # config_file = Path("config_daily.yaml")
        # config_file = Path("config_archive.yaml")
        config_file = Path("config_debug.yaml")
        logger.debug(f"Using config file: {config_file}")

        config = self.config
        logger.debug("Config loaded successfully")

        logger.debug("Getting Telethon client")
        client = await get_telethon_client()

        # 3. for each chat - pick a config that applies to it
        logger.debug("Starting chat processing loop...")

        for chat in tqdm(chats, desc="Processing chats"):
            logger.debug(f"Processing chat: {chat.name}")
            chat_config = self._pick_chat_config(chat, config)
            logger.debug(f"Selected config type: {type(chat_config).__name__}")

            # 4. download messages as per config
            logger.debug(f"Downloading messages for chat: {chat.name}")
            messages = await self._download_messages(client, chat, chat_config)
            logger.debug(f"Downloaded {len(messages)} messages")

            # 5. save messages to a database
            logger.debug(f"Saving {len(messages)} messages to database")
            if len(messages) > 0:
                self._save_messages(messages)
            logger.debug("Messages saved successfully")

    async def _load_chats(self) -> List[ChatData]:
        logger.debug("Starting chat loading process...")
        logger.debug("Getting raw chats from get_chats()")
        chats = await get_chats()
        logger.debug(f"Got {len(chats)} raw chats")

        res_chats = []
        logger.debug("Converting raw chats to ChatData objects")
        for chat in chats:
            # logger.debug(f"Converting chat: {chat['entity']}")
            res_chats.append(
                ChatData(entity=chat["entity"], last_message_date=chat["last_message_date"])
            )
        logger.debug(f"Converted {len(res_chats)} chats to ChatData objects")
        return res_chats

    # def _load_config(self, config_path: Path):
    #     logger.debug(f"Loading config from {config_path}")
    #     config = TelegramDownloaderConfig.from_yaml(config_path)
    #     logger.debug("Config loaded successfully")
    #     return config

    def _pick_chat_config(
        self, chat: ChatData, config: TelegramDownloaderConfig
    ) -> ChatCategoryConfig:
        logger.debug(f"Picking config for chat: {chat.name}")
        logger.debug(f"Chat category: {chat.entity_category}")

        if chat.entity_category == "group":
            if chat.is_owned:
                logger.debug("Selected: owned groups config")
                return config.owned_groups
            else:
                logger.debug("Selected: other groups config")
                return config.other_groups
        elif chat.entity_category == "channel":
            if chat.is_owned:
                logger.debug("Selected: owned channels config")
                return config.owned_channels
            else:
                logger.debug("Selected: other channels config")
                return config.other_channels
        elif chat.entity_category == "bot":
            logger.debug("Selected: bot config")
            return config.bots
        elif chat.entity_category == "private chat":
            logger.debug("Selected: private chat config")
            return config.private_chats
        else:
            raise ValueError(f"Invalid chat category: {chat.entity_category}")

    async def _download_messages(
        self, client: TelegramClient, chat: ChatData, chat_config: ChatCategoryConfig
    ):
        logger.debug(f"Starting message download for chat: {chat.name}")

        if not chat_config or not chat_config.enabled:
            logger.debug(f"Skipping chat {chat.name}: config disabled")
            return []

        logger.info(f"Downloading messages from chat: {chat.name}")

        # Initialize parameters for message download
        kwargs = {}
        logger.debug("Initializing download parameters")

        # Apply config parameters
        if chat_config.backdays:
            # Make min_date timezone-aware to match message.date
            min_date = datetime.now(timezone.utc) - timedelta(days=chat_config.backdays)
            logger.debug(f"Set min_date to {min_date}")
        else:
            min_date = None
            logger.debug("No min_date set")

        if chat_config.limit:
            kwargs["limit"] = chat_config.limit
            logger.debug(f"Set message limit to {chat_config.limit}")

        # Skip large groups if configured
        if chat_config.skip_big and chat.is_big:
            logger.warning(f"Skipping large chat {chat.name}")
            return []

        messages = await self._load_message_range(chat, min_date=min_date, **kwargs)

        logger.info(f"Downloaded {len(messages)} messages from {chat.name}")
        return messages

    async def get_telethon_client(self) -> TelegramClient:
        if self._telethon_client is None:
            self._telethon_client = await self._get_telethon_client()
        return self._telethon_client

    async def _load_message_range(
        self, chat: ChatData, min_date: datetime, **kwargs
    ) -> List[Message]:
        client = await self.get_telethon_client()

        messages = []
        try:
            logger.debug("Starting message iteration")
            async for message in atqdm(
                client.iter_messages(chat.entity, **kwargs),
                desc=f"Downloading messages for chat {chat.name}",
            ):

                # Skip messages older than min_date
                if min_date and message.date < min_date:
                    logger.debug(f"Reached message older than min_date ({message.date}), stopping")
                    break

                # Use built-in to_json() method
                messages.append(message)
                # logger.debug(f"Added message ID {message.id}")

        except Exception as e:
            logger.error(f"Error downloading messages from {chat.name}: {e}")
            import traceback

            logger.error("Full traceback:\n" + "".join(traceback.format_exc()))

            return []

        return messages

    def _get_database(self):
        # option 1 - mongodb pymongo client

        logger.info("Starting MongoDB setup...")

        # Get MongoDB connection string and database name from environment variables

        conn_str = self.env.MONGO_CONN_STR
        db_name = self.env.MONGO_DB_NAME

        logger.debug(f"Using database name: {db_name}")
        logger.debug("Attempting to connect to MongoDB...")

        # Connect to MongoDB
        client = MongoClient(conn_str)
        logger.info("Successfully connected to MongoDB")

        # MongoDB creates databases and collections automatically when you first store data
        # But we can explicitly create them to ensure they exist
        logger.debug("Checking if database exists...")
        if db_name not in client.list_database_names():
            logger.debug(f"Creating database: {db_name}")
            db = client[db_name]
        else:
            logger.debug(f"Using existing database: {db_name}")
            db = client[db_name]

        # Define collections we'll need
        collections = {
            "messages": "telegram_messages",
            "chats": "telegram_chats",
            "users": "telegram_users",
            "heartbeats": "telegram_heartbeats",
        }

        logger.debug("Starting collection setup...")

        # Create collections if they don't exist
        for purpose, collection_name in collections.items():
            logger.debug(f"Checking collection: {collection_name}")
            if collection_name not in db.list_collection_names():
                logger.debug(f"Creating collection: {collection_name}")
                db.create_collection(collection_name)
            else:
                logger.debug(f"Using existing collection: {collection_name}")

        logger.debug("Collection setup complete")

        # add item, read items - to the test heartbeats collection
        logger.debug("Testing heartbeats collection...")
        heartbeats_collection = db.heartbeats

        logger.debug("Inserting test heartbeat...")
        heartbeats_collection.insert_one({"timestamp": datetime.now()})
        logger.debug("Reading test heartbeat...")
        heartbeats_collection.find_one()
        logger.info(
            f"MongoDB setup complete. Using database '{db_name}' with collections: {', '.join(collections.values())}"
        )

        return db

    @property
    def db(self):
        if self._db is None:
            self._db = self._get_database()
        return self._db

    def _save_messages(self, messages):
        """Save messages to a database"""
        db = self.db
        logger.debug("Starting message save process")

        # step 1: get a db connection - i already did this somewhere
        logger.debug("Getting database connection")
        collection_name = "telegram_messages"
        collection = db[collection_name]
        logger.debug(f"Got collection: {collection_name}")

        # make messages to json format
        logger.debug("Converting messages to JSON")
        messages_json = [message.to_dict() for message in messages]
        logger.debug(f"Converted {len(messages_json)} messages to JSON")

        logger.debug("Inserting messages into database")

        collection.insert_many(messages_json)
        logger.debug("Messages inserted successfully")

    async def _get_telethon_client(self) -> TelegramClient:

        # Example initialization:
        SESSIONS_DIR = Path("sessions")
        SESSIONS_DIR.mkdir(exist_ok=True)
        TELEGRAM_API_ID = self.env.TELEGRAM_API_ID
        if TELEGRAM_API_ID is None:
            raise ValueError("TELEGRAM_API_ID is not set")
        TELEGRAM_API_ID = int(TELEGRAM_API_ID)
        TELEGRAM_API_HASH = self.env.TELEGRAM_API_HASH
        if TELEGRAM_API_HASH is None:
            raise ValueError("TELEGRAM_API_HASH is not set")
        telethon_manager = TelethonClientManager(
            storage_mode=StorageMode.TO_DISK,
            api_id=TELEGRAM_API_ID,
            api_hash=TELEGRAM_API_HASH,
            sessions_dir=SESSIONS_DIR,
        )

        user_id = self.env.TELEGRAM_USER_ID

        # Get client for user
        client = await telethon_manager.get_telethon_client(int(user_id))

        if not client:
            raise ValueError("Failed to get client")

        return client
