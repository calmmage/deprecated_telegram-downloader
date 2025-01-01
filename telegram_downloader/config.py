from enum import Enum
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class StorageMode(str, Enum):
    MONGO = "mongo"
    LOCAL = "local"


class TelegramDownloaderEnvSettings(BaseSettings):
    # Telegram API credentials
    TELEGRAM_API_ID: int  # | None = None
    TELEGRAM_API_HASH: str  # | None = None
    TELEGRAM_USER_ID: str = Field(default="291560340")

    TELETHON_SESSION_STR: str | None = None

    # Optional database connection if needed
    MONGO_CONN_STR: str | None = None
    MONGO_DB_NAME: str = Field(default="telegram-messages-dec-2024")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


class SizeThresholds(BaseModel):
    max_members_group: int = 1000  # groups larger than this are considered "big"
    max_members_channel: int = 1000  # channels larger than this are considered "big"


class ChatCategoryConfig(BaseModel):
    enabled: bool = False
    # message loading config
    backdays: Optional[int] = None
    limit: Optional[int] = None
    # todo: not implemented, use
    download_attachments: bool = False

    # chat filtering config
    whitelist: List[str] = Field(default_factory=list)
    blacklist: List[str] = Field(default_factory=list)
    skip_big: bool = True  # whether to skip large groups/channels


class TelegramDownloaderConfig(BaseModel):
    # Add storage_mode field at the top level
    storage_mode: StorageMode = Field(default=StorageMode.MONGO)

    size_thresholds: SizeThresholds = Field(default_factory=SizeThresholds)
    owned_groups: ChatCategoryConfig = Field(default_factory=ChatCategoryConfig)
    owned_channels: ChatCategoryConfig = Field(default_factory=ChatCategoryConfig)
    other_groups: ChatCategoryConfig = Field(default_factory=ChatCategoryConfig)
    other_channels: ChatCategoryConfig = Field(default_factory=ChatCategoryConfig)
    private_chats: ChatCategoryConfig = Field(default_factory=ChatCategoryConfig)
    bots: ChatCategoryConfig = Field(default_factory=ChatCategoryConfig)

    @classmethod
    def from_yaml(cls, path: Path, **kwargs) -> "TelegramDownloaderConfig":
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        config_dict.update(kwargs)
        return cls(**config_dict)
