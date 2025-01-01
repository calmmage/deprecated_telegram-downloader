import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from loguru import logger
from pymongo import MongoClient

from telegram_downloader.data_model import ChatData


class TelegramDB:
    def __init__(self):
        load_dotenv()
        self.conn_str = os.getenv("MONGO_CONN_STR")
        self.db_name = os.getenv("MONGO_DB_NAME")

        # Add collection names from env
        self.messages_collection = os.getenv("MONGO_MESSAGES_COLLECTION", "telegram_messages")
        self.chats_collection = os.getenv("MONGO_CHATS_COLLECTION", "telegram_chats")
        self.users_collection = os.getenv("MONGO_USERS_COLLECTION", "telegram_users")
        self.heartbeats_collection = os.getenv("MONGO_HEARTBEATS_COLLECTION", "telegram_heartbeats")
        self.app_data_collection = os.getenv(
            "MONGO_APP_DATA_COLLECTION", "telegram_downloader_app_data"
        )

        self._client = None
        self._db = None

    @property
    def client(self):
        if self._client is None:
            self._client = MongoClient(self.conn_str)
        return self._client

    @property
    def db(self):
        if self._db is None:
            self._db = self.client[self.db_name]
        return self._db

    def get_chats(self) -> List[ChatData]:
        """Get all chats from database"""
        return [ChatData(**item) for item in self.db[self.chats_collection].find()]

    def find_chat(self, query: dict) -> Optional[ChatData]:
        """Find a single chat matching the query"""
        result = self.db[self.chats_collection].find_one(query)
        return ChatData(**result) if result else None

    def get_messages(self, chat_id: int):
        """Get all messages for a specific chat"""
        return list(self.db[self.messages_collection].find({"chat_id": chat_id}))

    def find_message(self, query: dict):
        """Find a single message matching the query"""
        return self.db[self.messages_collection].find_one(query)
