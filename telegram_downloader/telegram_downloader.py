import asyncio
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

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
        self.env = TelegramDownloaderEnvSettings(**kwargs)

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

        logger.debug("Setting up logger...")
        setup_logger(logger, level="DEBUG" if args.debug else "INFO")

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
        # 1. load list of chats
        logger.debug("Loading chats...")
        chats = await self._get_chats()
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

        # 3. for each chat - pick a config that applies to it
        logger.debug("Starting chat processing loop...")

        for chat in tqdm(chats, desc="Processing chats"):
            logger.debug(f"Processing chat: {chat.name}")
            chat_config = self._pick_chat_config(chat)
            logger.debug(f"Selected config type: {type(chat_config).__name__}")

            # 4. download messages as per config
            logger.debug(f"Downloading messages for chat: {chat.name}")
            messages = await self._download_messages(chat, chat_config)
            logger.debug(f"Downloaded {len(messages)} messages")

            # 5. save messages to a database
            logger.debug(f"Saving {len(messages)} messages to database")
            if len(messages) > 0:
                self._save_messages(messages)
            logger.debug("Messages saved successfully")

    @property
    def storage_mode(self):
        return self.config.storage_mode

    async def _get_chat_list(self, client=None) -> List[Dialog]:
        logger.debug("Getting chat list from Telegram")
        if client is None:
            client = await self.get_telethon_client()
        chats = await client.get_dialogs()
        logger.debug(f"Retrieved {len(chats)} dialogs from Telegram")
        return chats

    # ✅
    async def _get_chats_from_client(self) -> List[ChatData]:
        logger.debug("Getting chats from client asynchronously")
        # Store chats in memory
        client = await self.get_telethon_client()
        dialogs = await client.get_dialogs()

        logger.debug(f"Retrieved {len(dialogs)} chats")
        result = []
        for dialog in dialogs:
            if type(dialog.entity).__name__ == "ChatForbidden":
                logger.warning(f"Skipping 'forbidden chat': {dialog.entity}")
                continue
            chat = ChatData(entity=dialog.entity, last_message_date=dialog.date)
            result.append(chat)
        return result

    # ✅
    def _has_fresh_chats_on_disk(self, max_age: timedelta = timedelta(days=1)):
        logger.warning("Outdated code - disk storage. Needs to be updaged")
        # todo: use self.chat_refresh_timestamp
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

    # ✅
    def _save_chats_to_disk(self, chats):
        logger.warning("Outdated code - disk storage. Needs to be updaged")
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

    # ✅
    def _save_chats_to_db(self, chats: List[ChatData], collection_name: str = "telegram_chats"):
        # todo: rework to Motor
        for chat in chats:
            self.db[collection_name].update_one(
                {"id": chat.id},  # Use id as the key for upsert
                {"$set": chat.to_dict()},
                upsert=True,
            )

        logger.debug(f"Upserted {len(chats)} chats to db")

    # ✅
    def _load_chats_from_db(self, collection_name: str = "telegram_chats") -> List[ChatData]:
        telegram_chats_collection = self.db[collection_name]
        chats = [ChatData(**item) for item in telegram_chats_collection.find()]
        logger.info(f"Loaded {len(chats)} entities from db collection {collection_name}")
        return chats

    # ✅
    def _has_fresh_chats_in_db(self, max_age: timedelta = timedelta(days=1)) -> bool:
        """Check if the chats in the database are fresh (recently updated).

        Args:
            collection_name (str, optional): The name of the collection to check. Defaults to "chats".
            max_age (timedelta, optional): The maximum age for chats to be considered fresh.
                Defaults to 1 day.

        Returns:
            bool: True if chats are fresh, False otherwise.
        """
        timestamp = self.chat_refresh_timestamp
        if timestamp is None:
            return False

        now = datetime.now(timezone.utc)
        is_fresh = now - timestamp < max_age
        logger.debug(f"Chats in db are {'fresh' if is_fresh else 'stale'}")
        return is_fresh

    # ✅
    @property
    def chat_refresh_timestamp(self) -> datetime | None:
        """Get the timestamp of the last chat refresh operation.

        Returns:
            datetime | None: The UTC timestamp of the last chat refresh, or None if no refresh has occurred.
            The returned datetime is always timezone-aware in UTC.

        Raises:
            ValueError: If the storage mode is invalid or not supported.
            NotImplementedError: If local storage mode is used (not yet implemented).
        """
        if self.storage_mode == StorageMode.MONGO:
            return self._get_chat_refresh_timestamp_from_db()
        elif self.storage_mode == StorageMode.LOCAL:
            raise NotImplementedError("Local storage not implemented for chat_refresh_timestamp")
        else:
            raise ValueError(f"Invalid storage mode: {self.storage_mode}")

    # ✅
    def _get_chat_refresh_timestamp_from_db(self) -> datetime | None:
        """Retrieve the chat refresh timestamp from MongoDB.

        Returns:
            datetime | None: The UTC timestamp of the last chat refresh, or None if no refresh has occurred.
            The returned datetime is always timezone-aware in UTC.
        """
        collection = self.db["telegram_downloader_app_data"]
        data = collection.find_one({"key": "chat_refresh_timestamp"})
        if data and "value" in data:
            timestamp = data["value"]
            # Ensure the timestamp is timezone-aware in UTC
            if isinstance(timestamp, datetime):
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                else:
                    timestamp = timestamp.astimezone(timezone.utc)
                return timestamp
            return None
        return None

    # ✅
    def _save_chat_refresh_timestamp_to_db(self, timestamp: datetime | None) -> None:
        """Save the chat refresh timestamp to MongoDB.

        Args:
            timestamp (datetime | None): The timestamp to save. If provided, must be timezone-aware.
                Will be converted to UTC before saving.

        Raises:
            ValueError: If the timestamp is naive (has no timezone information).
        """
        if timestamp is not None and timestamp.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware")

        if timestamp is not None:
            timestamp = timestamp.astimezone(timezone.utc)

        collection = self.db["telegram_downloader_app_data"]
        collection.update_one(
            {"key": "chat_refresh_timestamp"},
            {"$set": {"value": timestamp}},
            upsert=True,
        )

    # ✅
    @chat_refresh_timestamp.setter
    def chat_refresh_timestamp(self, value: datetime | None) -> None:
        """Set the timestamp of the last chat refresh operation.

        Args:
            value (datetime | None): The timestamp to set. If provided, must be timezone-aware.
                Will be converted to UTC before saving.

        Raises:
            ValueError: If the storage mode is invalid or if the timestamp is naive.
            NotImplementedError: If local storage mode is used (not yet implemented).
        """
        if value is not None and value.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware")

        if self.storage_mode == StorageMode.MONGO:
            self._save_chat_refresh_timestamp_to_db(value)
        elif self.storage_mode == StorageMode.LOCAL:
            raise NotImplementedError("Local storage not implemented for chat_refresh_timestamp")
        else:
            raise ValueError(f"Invalid storage mode: {self.storage_mode}")

    # ✅
    async def _get_chats(self) -> List[ChatData]:
        logger.debug("Getting chats asynchronously")
        if self.storage_mode == StorageMode.MONGO:
            if self._has_fresh_chats_in_db():  # ✅
                logger.debug("Found fresh chats in database")
                return self._load_chats_from_db()  # ✅
            else:
                logger.debug("No fresh chats found, getting from client")
                chats = await self._get_chats_from_client()  # ✅
                self._save_chats_to_db(chats)  # ✅
                return chats
        elif self.storage_mode == StorageMode.LOCAL:
            if self._has_fresh_chats_on_disk():  # ✅
                logger.debug("Found fresh chats on disk")
                return self._get_chats_from_disk()  # ✅
            else:
                logger.debug("No fresh chats found, getting from client")
                chats = await self._get_chats_from_client()  # ✅
                self._save_chats_to_disk(chats)  # ✅
                return chats
        else:
            raise ValueError(f"Invalid storage mode: {self.storage_mode}")

    # Create constructor map
    # ✅
    @staticmethod
    def _load_chats_from_json(filepath):
        logger.warning(
            "Using outdated method to load chats from JSON - needs to be reworked, see the code comments"
        )
        # todo: use new ChatData class 'from_json' feature
        # todo: rework to just using to_dict feature of entity class
        # type_map = {
        #     "Channel": lambda d: Channel(**d),
        #     "User": lambda d: User(**d),
        #     "Chat": lambda d: Chat(**d),
        # }
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        chats = []
        for item in data["chats"]:
            chat = ChatData.from_json(item)
            if chat is None:
                logger.warning(f"Skipping 'forbidden chat': {item}")
                continue
            chats.append(chat)

        logger.info(f"Loaded {len(chats)} entities from {filepath}")
        return chats

    # ✅
    def _get_chats_from_disk(self):
        logger.warning("Outdated code - disk storage. Needs to be updated")
        logger.debug("Loading chats from disk")
        data_path = Path("data/chats.json")

        chats = self._load_chats_from_json(data_path)

        logger.debug(f"Loaded {len(chats)} chats from disk")
        return chats

    # ✅
    def _pick_chat_config(self, chat: ChatData) -> ChatCategoryConfig:
        logger.debug(f"Picking config for chat: {chat.name}")
        logger.debug(f"Chat category: {chat.entity_category}")

        if chat.entity_category == "group":
            if chat.is_owned:
                logger.debug("Selected: owned groups config")
                return self.config.owned_groups
            else:
                logger.debug("Selected: other groups config")
                return self.config.other_groups
        elif chat.entity_category == "channel":
            if chat.is_owned:
                logger.debug("Selected: owned channels config")
                return self.config.owned_channels
            else:
                logger.debug("Selected: other channels config")
                return self.config.other_channels
        elif chat.entity_category == "bot":
            logger.debug("Selected: bot config")
            return self.config.bots
        elif chat.entity_category == "private chat":
            logger.debug("Selected: private chat config")
            return self.config.private_chats
        else:
            raise ValueError(f"Invalid chat category: {chat.entity_category}")

    async def _download_messages(self, chat: ChatData, chat_config: ChatCategoryConfig):
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
            storage_mode=StorageMode.LOCAL,
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
