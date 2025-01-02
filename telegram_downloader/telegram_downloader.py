import json
import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from re import M
from typing import List

from loguru import logger
from pymongo import MongoClient
from telethon import TelegramClient
from telethon.sessions import StringSession
from telethon.types import Dialog, Message
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from telegram_downloader.config import (
    ChatCategoryConfig,
    TelegramDownloaderConfig,
    TelegramDownloaderEnvSettings,
)
from telegram_downloader.data_model import ChatData
from telegram_downloader.telethon_client_manager import StorageMode, TelethonClientManager
from telegram_downloader.utils import ensure_utc_datetime, setup_logger


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
        self.reset_properties()

    async def main(self, ignore_finished=False, chat_sample_size=None):
        # 1. load list of chats
        logger.debug("Loading chats...")
        chats = await self._get_chats()
        logger.info(f"Loaded {len(chats)} chats")

        if chat_sample_size is not None:
            random.shuffle(chats)
            chats = chats[:chat_sample_size]
            logger.info(f"Limited to {len(chats)} chats")

        # 3. for each chat - pick a config that applies to it
        logger.debug("Starting chat processing loop...")

        for chat in tqdm(chats, desc="Processing chats"):
            logger.debug(f"Processing chat: {chat.name}")
            chat_config = self._pick_chat_config(chat)
            logger.debug(f"Selected config type: {type(chat_config).__name__}")

            # 4. download messages as per config
            logger.info(f"Downloading messages for chat: {chat.name}")
            messages = await self._download_messages(chat, chat_config, ignore_finished)
            if messages:
                min_date = min(msg.date for msg in messages).strftime("%d %b %Y")
                max_date = max(msg.date for msg in messages).strftime("%d %b %Y")
                logger.info(f"Downloaded {len(messages)} messages: {min_date} - {max_date}")
            else:
                logger.info("Downloaded 0 messages")

            # 5. save messages to a database
            logger.debug(f"Saving {len(messages)} messages to database")
            if len(messages) > 0:
                self._save_messages(messages)
            logger.debug("Messages saved successfully")

        # Add duplicate cleanup at the end
        logger.info("Starting duplicate message cleanup")
        deleted_count = self.delete_duplicate_messages()
        logger.info(f"Duplicate cleanup complete. Deleted {deleted_count} messages")

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
        result = self._filter_redundant_chats(result)
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
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        data_path = Path("data/chats.json")
        data_path.parent.mkdir(exist_ok=True)
        data_path.write_text(json.dumps(data, indent=2))
        logger.debug("Chats saved successfully")

    # ✅
    def _save_chats_to_db(self, chats: List[ChatData]):
        for chat in chats:
            self.chats_collection.update_one(
                {"id": chat.id},  # Use id as the key for upsert
                {"$set": chat.to_dict()},
                upsert=True,
            )
        logger.debug(f"Upserted {len(chats)} chats to db")

    # ✅
    def _load_chats_from_db(self) -> List[ChatData]:
        """Load chats from database and verify the refresh timestamp."""

        chats = []
        for item in self.chats_collection.find():
            item.pop("_id")
            chat = ChatData(**item)
            chats.append(chat)

        logger.info(f"Loaded {len(chats)} chats from database")
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
        collection = self.db[self.env.MONGO_APP_DATA_COLLECTION]
        data = collection.find_one({"key": "chat_refresh_timestamp"})
        if data and "value" in data:
            return ensure_utc_datetime(data["value"])
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

        collection = self.db[self.env.MONGO_APP_DATA_COLLECTION]
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
            if self._has_fresh_chats_in_db():
                logger.debug("Found fresh chats in database")
                return self._load_chats_from_db()
            else:
                logger.debug("No fresh chats found, getting from client")
                chats = await self._get_chats_from_client()
                self._save_chats_to_db(chats)
                # Set the refresh timestamp after successfully saving chats
                self.chat_refresh_timestamp = datetime.now(timezone.utc)
                return chats
        elif self.storage_mode == StorageMode.LOCAL:
            if self._has_fresh_chats_on_disk():
                logger.debug("Found fresh chats on disk")
                return self._get_chats_from_disk()  # ✅
            else:
                logger.debug("No fresh chats found, getting from client")
                chats = await self._get_chats_from_client()
                self._save_chats_to_disk(chats)
                # Set the refresh timestamp after successfully saving chats
                self.chat_refresh_timestamp = datetime.now(timezone.utc)
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

    def _get_chat_message_range(self, chat_id: int) -> tuple[datetime, datetime]:
        """Get the min and max timestamps of messages for a chat in the database.

        Args:
            chat_id: The ID of the chat to check

        Returns:
            tuple[datetime | None, datetime | None]: (min_timestamp, max_timestamp)
            Returns (None, None) if no messages found. Returned datetimes are timezone-aware (UTC).
        """
        logger.debug(f"Getting message range for chat {chat_id}")
        collection = self.messages_collection

        chat_data = self.chats[chat_id]

        if type(chat_data.entity).__name__ == "Channel":
            query = {"peer_id": {"_": "PeerChannel", "channel_id": chat_id}}
        elif type(chat_data.entity).__name__ == "User":
            query = {"peer_id": {"_": "PeerUser", "user_id": chat_id}}
        elif type(chat_data.entity).__name__ == "Chat":
            query = {"peer_id": {"_": "PeerChat", "chat_id": chat_id}}
        else:
            raise ValueError(f"Unknown chat type: {chat_data.entity_category}")

        # Find min timestamp
        min_result = collection.find(query, {"date": 1}).sort("date", 1).limit(1)

        # Find max timestamp
        max_result = collection.find(query, {"date": 1}).sort("date", -1).limit(1)

        min_doc = next(min_result, None)
        max_doc = next(max_result, None)

        # Ensure timestamps are timezone-aware
        min_timestamp = ensure_utc_datetime(min_doc["date"] if min_doc else None)
        max_timestamp = ensure_utc_datetime(max_doc["date"] if max_doc else None)

        # Convert naive datetimes to timezone-aware if needed
        if min_timestamp and min_timestamp.tzinfo is None:
            min_timestamp = min_timestamp.replace(tzinfo=timezone.utc)
        if max_timestamp and max_timestamp.tzinfo is None:
            max_timestamp = max_timestamp.replace(tzinfo=timezone.utc)

        logger.debug(f"Message range for chat {chat_id}: {min_timestamp} to {max_timestamp}")
        return min_timestamp, max_timestamp

    async def _download_messages(
        self, chat: ChatData, chat_config: ChatCategoryConfig, ignore_finished: bool = False
    ):
        logger.debug(f"Starting message download for chat: {chat.name}")

        if not chat_config or not chat_config.enabled:
            logger.debug(f"Skipping chat {chat.name}: config disabled")
            return []

        logger.info(f"Downloading messages from chat: {chat.name}")

        messages = []
        try:
            if chat_config.backdays:
                config_min_date = datetime.now(timezone.utc) - timedelta(days=chat_config.backdays)
                last_message_date = ensure_utc_datetime(chat.last_message_date)

                if last_message_date and last_message_date < config_min_date:
                    logger.info(
                        f"Skipping chat {chat.name}: last message ({last_message_date}) "
                        f"is older than minimum date ({config_min_date})"
                    )
                    # Mark as finished since all messages are too old
                    chat.finished_downloading = True
                    self._save_chat_to_db(chat)
                    return []

            if chat_config.skip_big and chat.is_big:
                logger.info(f"Skipping chat {chat.name}: big chat")
                return []

            # Get existing message range
            min_timestamp, max_timestamp = self._get_chat_message_range(chat.id)

            if min_timestamp is None:
                # Skip if chat is marked as finished downloading
                if chat.finished_downloading and not ignore_finished:
                    logger.info(f"Skipping chat {chat.name}: marked as finished")
                    return []
                logger.info("No existing messages found, performing full download")
                messages = await self._download_all_messages(chat, chat_config)
            else:
                logger.info(
                    f"Already have messages for chat {chat.name} from {min_timestamp} to {max_timestamp}"
                )
                messages = await self._download_messages_excluding_range(
                    chat, chat_config, min_timestamp, max_timestamp, ignore_finished=ignore_finished
                )

            # After successful download, mark as finished
            chat.finished_downloading = True
            self._save_chat_to_db(chat)
            return messages

        except Exception as e:
            logger.error(f"Error downloading messages from {chat.name}: {e}")
            # Don't mark as finished if there was an error
            return []

    async def _download_all_messages(self, chat: ChatData, chat_config: ChatCategoryConfig):
        kwargs = {}
        if chat_config.backdays:
            config_min_date = datetime.now(timezone.utc) - timedelta(days=chat_config.backdays)
            kwargs["offset_date"] = config_min_date
            kwargs["reverse"] = True

        if chat_config.limit:
            kwargs["limit"] = chat_config.limit

        return await self._load_messages(chat, **kwargs)

    async def _download_messages_excluding_range(
        self,
        chat: ChatData,
        chat_config: ChatCategoryConfig,
        min_timestamp: datetime,
        max_timestamp: datetime,
        ignore_finished: bool = False,
    ):

        logger.info(f"Getting messages newer than {max_timestamp}")
        # part 1: download newer messages
        messages = await self._load_messages(chat, offset_date=max_timestamp, reverse=True)
        logger.info(f"Downloaded {len(messages)} newer messages")

        logger.info(f"Getting messages older than {min_timestamp}")
        # part 2: download older messages

        # Skip if chat is marked as finished downloading
        if chat.finished_downloading and not ignore_finished:
            logger.info(f"Skipping the older messages - chat is marked as finished")
            return messages

        messages.extend(
            await self._load_messages(
                chat, offset_date=min_timestamp, reverse=False, limit=chat_config.limit
            )
        )
        # logger.info(f"Downloaded {len(messages)} messages total")

        return messages

    async def get_telethon_client(self) -> TelegramClient:
        if self._telethon_client is None:
            if self.env.TELETHON_SESSION_STR:
                self._telethon_client = TelegramClient(
                    StringSession(self.env.TELETHON_SESSION_STR),
                    api_id=self.env.TELEGRAM_API_ID,
                    api_hash=self.env.TELEGRAM_API_HASH,
                )
                await self._telethon_client.start()
            else:
                self._telethon_client = await self._get_telethon_client()
        return self._telethon_client

    async def _load_messages(
        self, chat: ChatData, offset_date: datetime | None = None, reverse: bool = False, **kwargs
    ) -> List[Message]:
        """Load messages from Telegram.

        Args:
            chat: Chat to load messages from
            offset_date: Date to start loading messages from
            reverse: If True, load newer messages (from offset_date forward)
                    If False, load older messages (from offset_date backward)
            **kwargs: Additional arguments for iter_messages
        """
        client = await self.get_telethon_client()
        messages = []

        try:
            logger.debug(
                f"Loading messages for {chat.name} {'after' if reverse else 'before'} {offset_date}"
            )
            async for message in atqdm(
                client.iter_messages(
                    chat.entity, offset_date=offset_date, reverse=reverse, **kwargs
                ),
                desc=f"Downloading messages for chat {chat.name}",
            ):
                messages.append(message)

        except Exception as e:
            logger.error(f"Error downloading messages from {chat.name}: {e}")
            import traceback

            logger.error("Full traceback:\n" + "".join(traceback.format_exc()))

            return []

        return messages

    def _get_database(self):
        logger.info("Starting MongoDB setup...")

        # Get MongoDB connection string and database name from environment variables
        conn_str = self.env.MONGO_CONN_STR
        db_name = self.env.MONGO_DB_NAME

        logger.debug(f"Using database name: {db_name}")
        logger.debug("Attempting to connect to MongoDB...")

        # Connect to MongoDB
        client = MongoClient(conn_str)
        logger.info("Successfully connected to MongoDB")

        logger.debug("Checking if database exists...")
        if db_name not in client.list_database_names():
            logger.debug(f"Creating database: {db_name}")
            db = client[db_name]
        else:
            logger.debug(f"Using existing database: {db_name}")
            db = client[db_name]

        # Define collections using env settings
        collections = {
            "messages": self.env.MONGO_MESSAGES_COLLECTION,
            "chats": self.env.MONGO_CHATS_COLLECTION,
            "users": self.env.MONGO_USERS_COLLECTION,
            "heartbeats": self.env.MONGO_HEARTBEATS_COLLECTION,
            "app_data": self.env.MONGO_APP_DATA_COLLECTION,
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

        # Test heartbeats collection
        logger.debug("Testing heartbeats collection...")
        heartbeats_collection = db[self.env.MONGO_HEARTBEATS_COLLECTION]

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

    def _save_messages(self, messages: List[Message]):
        """Save messages to a database"""
        if self.storage_mode == StorageMode.MONGO:
            logger.debug("Starting message save process")

            logger.debug("Getting database connection")
            collection = self.messages_collection

            # make messages to json format
            logger.debug("Converting messages to JSON")
            messages_json = []
            for message in messages:
                message_dict = message.to_dict()
                # Extract chat_id using existing method
                try:
                    chat_id = self._get_message_chat_id_from_raw(message_dict)
                    message_dict["chat_id"] = chat_id
                    messages_json.append(message_dict)
                except ValueError as e:
                    logger.warning(f"Skipping message - {e}")
                    continue

            logger.debug(f"Converted {len(messages_json)} messages to JSON")

            logger.debug("Inserting messages into database")
            collection.insert_many(messages_json)
            logger.debug("Messages inserted successfully")
        else:
            raise ValueError(f"Invalid storage mode: {self.storage_mode}")

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

    # region Chat stats

    def get_random_chat(
        self,
        chats: List[ChatData],
        entity_type: str = None,  # 'group', 'channel', 'bot', 'private chat'
        owned: bool = None,
        recent: bool = None,
        big: bool = None,
        recent_threshold: timedelta = timedelta(days=30),
        big_threshold: int = 1000,
    ):
        if entity_type is not None:
            chats = [chat for chat in chats if chat.entity_category == entity_type]
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
            chats = [chat for chat in chats if chat.is_recent == recent]
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

    def calculate_stats(
        self,
        chats: List[ChatData],
        big_threshold: int = 1000,
        recent_threshold: timedelta = timedelta(days=30),
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
            entity = chat.entity
            category = chat.entity_category
            stats[category + "s"] += 1

            # last_message_date = datetime.fromisoformat(getattr(chat, 'date', None)) if getattr(chat, 'date', None) else None
            is_recent = chat.get_is_recent(recent_threshold)
            if is_recent:
                stats[f"recent_{category}s"] += 1

            owner = getattr(entity, "creator", False)
            if owner:
                stats[f"owned_{category}s"] += 1
                if is_recent:
                    stats[f"recent_owned_{category}s"] += 1

            members = getattr(entity, "participants_count", 0)
            if members > big_threshold:
                stats[f"big_{category}s"] += 1

        # Bonus: size / time-based stats
        # for each of the above - count which have recent messages (last 30 days)
        logger.debug(f"Calculated stats: {stats}")
        return stats

    async def demo_chat_stats(self, debug: bool = False):
        setup_logger(logger, level="DEBUG" if debug else "INFO")
        logger.debug("Starting main function")
        chats = await self._get_chats()
        logger.debug("Main function completed")
        logger.debug(f"Loaded {len(chats)} chats")

        stats = self.calculate_stats(chats)
        logger.info("Chat statistics:")
        for key, value in stats.items():
            logger.info(f"{key}: {value}")

        random_chat = self.get_random_chat(chats, "group", recent=True, big=False)
        logger.info(f"Random recent group: {random_chat['entity'].title}")

        random_chat = self.get_random_chat(chats, "group", recent=False, big=False)
        logger.info(f"Random old group: {random_chat['entity'].title}")

        random_chat = self.get_random_chat(chats, "channel", big=False)
        logger.info(f"Random small channel: {random_chat['entity'].title}")

        random_chat = self.get_random_chat(chats, "channel", big=True)
        logger.info(f"Random big channel: {random_chat['entity'].title}")

        random_chat = self.get_random_chat(chats, "bot")
        logger.info(f"Random bot: {random_chat['entity'].username}")

        random_chat = self.get_random_chat(chats, "private chat", recent=True)
        logger.info(f"Random private chat: {random_chat['entity'].username}")

        random_chat = self.get_random_chat(chats, "private chat", recent=False)
        logger.info(f"Random old private chat: {random_chat['entity'].username}")

    # endregion chat stats

    # region plot_stats
    def plot_size_distribution(self, items, title, output_path: Path):

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

    def plot_small_size_distribution(self, items, title, output_path: Path):

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

    # region access utils
    @property
    def messages_raw(self):
        if self._messages_raw is None:
            self._messages_raw = list(self.messages_collection.find())
        return self._messages_raw

    @property
    def messages_collection(self):
        return self.db[self.env.MONGO_MESSAGES_COLLECTION]

    @property
    def chats_raw(self):
        if self._chats_raw is None:
            self._chats_raw = list(self.chats_collection.find())
        return self._chats_raw

    @property
    def chats_collection(self):
        return self.db[self.env.MONGO_CHATS_COLLECTION]

    @property
    def chats(self):
        if self._chats is None:
            self._chats = {}
            for item in self.chats_raw:
                item.pop("_id")
                self._chats[item["id"]] = ChatData(**item)
        return self._chats

    @property
    def messages(self):
        if self._messages is None:
            # todo: drop chat_id before init
            self._messages = []
            for item in self.messages_raw:
                item.pop("chat_id")
                self._messages.append(Message(**item))
        return self._messages

    def _get_message_chat_id(self, message: Message):
        try:
            return message.peer_id.channel_id
        except:
            pass
        try:
            return message.peer_id.user_id
        except:
            pass
        try:
            return message.peer_id.chat_id
        except:
            pass
        raise ValueError(f"Unknown peer_id type: {message.peer_id}")

    def _get_message_chat_id_from_raw(self, message: dict):
        peer_id = message.get("peer_id", {})
        if peer_id.get("_") == "PeerChannel":
            chat_id = peer_id.get("channel_id")
        elif peer_id.get("_") == "PeerUser":
            chat_id = peer_id.get("user_id")
        elif peer_id.get("_") == "PeerChat":
            chat_id = peer_id.get("chat_id")
        else:
            raise ValueError(f"Unknown peer_id type: {peer_id.get('_')}")
        return chat_id

    def reset_properties(self):
        self._messages = None
        self._messages_raw = None
        self._chats = None
        self._chats_raw = None
        self._chat_names = None

    # step 2
    @property
    def chat_names(self):
        if self._chat_names is None:
            self._chat_names = {chat_id: chat.name for chat_id, chat in self.chats.items()}
        return self._chat_names

    # chat_names = {}
    # for chat_id, chat in chats.items():
    #     chat_names[chat_id] = chat.name

    # step 3
    def find_chat(self, key):
        res = {}
        for chat_id, chat_name in self.chat_names.items():
            if key in chat_name:
                res[chat_id] = chat_name
        return res

    # chats_1 = find_chat("Лавровы и")
    # chats_1
    # len(chats_1)

    def resolve_chat(self, key):
        if key in self.chat_names:
            return key

        candidates = self.find_chat(key)
        if len(candidates) == 1:
            return list(candidates.keys())[0]
        else:
            raise ValueError(f"Multiple candidates found for {key}: {candidates}")

    # resolve_chat("Лавровы и Кремер")

    def get_chat_messages(self, chat_key):
        # todo: load newer messages from the client
        chat_id = self.resolve_chat(chat_key)
        chat_data = self.chats[chat_id]
        if type(chat_data.entity).__name__ == "Channel":
            query = {"peer_id": {"_": "PeerChannel", "channel_id": chat_id}}
        elif type(chat_data.entity).__name__ == "User":
            query = {"peer_id": {"_": "PeerUser", "user_id": chat_id}}
        elif type(chat_data.entity).__name__ == "Chat":
            query = {"peer_id": {"_": "PeerChat", "chat_id": chat_id}}
        else:
            raise ValueError(f"Unknown chat type: {chat_data.entity_category}")
        messages = self.messages_collection.find(query)
        messages = list(messages)
        return messages

    def _convert_chats_to_df(self, raw_chats):
        # step 1: flatten the data

        # step 2: create pandas df

        # todo: step 3 - filter out unwanted columns
        pass

    def _convert_messages_to_df(self, raw_messages):
        # step 1: flatten the data

        # step 2: create pandas df

        # todo: step 3 - filter out unwanted columns
        pass

    # get_chat_messages("Лавровы и Кремер")
    # m = get_chat_messages(893942447)
    # m = get_chat_messages(1585809155)

    # m = get_chat_messages(1125045120)

    # chat_names[1125045120]

    # item["peer_id"]
    # {'_': 'PeerUser', 'user_id': 322324998}

    # endregion access utils

    def _filter_redundant_chats(self, chats: List[ChatData]) -> List[ChatData]:
        """
        Filter out redundant chat entries, keeping Channel versions and handling migrated chats.

        Args:
            chats: List of ChatData objects to filter

        Returns:
            dict[int, ChatData]: Filtered dictionary of chat_id to ChatData
        """
        # First pass: build migration mapping and collect all chats
        migration_map = {}  # old_chat_id -> new_channel_id
        chat_map = {}  # chat_id -> ChatData

        for chat in chats:
            chat_map[chat.id] = chat

            # Check if this is a migrated chat
            migrated_to = getattr(chat.entity, "migrated_to", None)
            if migrated_to and hasattr(migrated_to, "channel_id"):
                migration_map[chat.id] = migrated_to.channel_id

        # Second pass: filter out migrated chats
        filtered_chats = []
        for chat_id, chat in chat_map.items():
            # Skip if this chat was migrated and we have the new channel
            if chat_id in migration_map:
                new_channel_id = migration_map[chat_id]
                if new_channel_id in chat_map:
                    logger.debug(
                        f"Skipping migrated chat {chat_id} ('{chat.name}') "
                        f"in favor of channel {new_channel_id}"
                    )
                    continue

            # Keep this chat
            filtered_chats.append(chat)

        return filtered_chats

    def _save_chat_to_db(self, chat: ChatData):
        """Update chat data in database."""
        self.chats_collection.update_one({"id": chat.id}, {"$set": chat.to_dict()}, upsert=True)

    def _find_duplicate_messages(self) -> tuple[dict, list]:
        """Find messages that are complete duplicates (all fields match except _id).

        Returns:
            tuple[dict, list]: (duplicates_by_id, duplicate_ids_to_delete)
                duplicates_by_id: Dict mapping message_id to list of duplicate messages
                duplicate_ids_to_delete: List of MongoDB _id values to delete
        """
        logger.info("Starting duplicate message detection")

        # Step 1: Group messages by id
        messages_by_id = defaultdict(list)
        for message in self.messages_collection.find():
            messages_by_id[message["id"]].append(message)

        # Step 2: Find duplicates
        duplicates = {}
        to_delete = []

        for message_id, messages in tqdm(messages_by_id.items(), desc="Checking for duplicates"):
            if len(messages) > 1:
                duplicates[message_id] = messages
                # Compare messages without _id field
                items_to_compare = []
                for message in messages:
                    _id = message.pop("_id")
                    duplicate = False
                    for item in items_to_compare:
                        if item == message:
                            duplicate = True
                            to_delete.append(_id)
                            break
                    if not duplicate:
                        items_to_compare.append(message)

        logger.info(f"Found {len(duplicates)} message IDs with potential duplicates")
        logger.info(f"Identified {len(to_delete)} messages for deletion")

        return duplicates, to_delete

    def delete_duplicate_messages(self) -> int:
        """Delete duplicate messages from the database.

        Returns:
            int: Number of deleted messages
        """
        _, to_delete = self._find_duplicate_messages()

        if not to_delete:
            logger.info("No duplicate messages found")
            return 0

        # Delete duplicates
        result = self.messages_collection.delete_many({"_id": {"$in": to_delete}})
        deleted_count = result.deleted_count

        logger.info(f"Deleted {deleted_count} duplicate messages")

        # Reset cached messages
        self.reset_properties()

        return deleted_count
