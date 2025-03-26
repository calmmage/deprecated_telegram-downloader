import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
import yaml

from telegram_downloader.config import (
    ChatCategoryConfig,
    SizeThresholds,
    StorageMode,
    TelegramDownloaderConfig,
    TelegramDownloaderEnvSettings,
)


@pytest.fixture
def sample_config_dict():
    return {
        "storage_mode": "mongo",
        "size_thresholds": {
            "max_members_group": 1500,
            "max_members_channel": 2000,
        },
        "owned_groups": {
            "enabled": True,
            "backdays": 30,
            "limit": 1000,
            "whitelist": ["group1", "group2"],
            "blacklist": ["badgroup"],
            "download_attachments": True,
            "skip_big": False,
        },
        "owned_channels": {
            "enabled": True,
            "backdays": 90,
            "limit": 500,
            "whitelist": [],
            "blacklist": [],
            "download_attachments": False,
            "skip_big": True,
        },
        "other_groups": {
            "enabled": False,
        },
        "other_channels": {
            "enabled": False,
        },
        "private_chats": {
            "enabled": True,
            "backdays": 365,
        },
        "bots": {
            "enabled": False,
        },
    }


@pytest.fixture
def sample_config_file(sample_config_dict):
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
        yaml.dump(sample_config_dict, temp_file)
        temp_path = Path(temp_file.name)
    
    yield temp_path
    
    # Clean up
    if temp_path.exists():
        temp_path.unlink()


def test_storage_mode_enum():
    assert StorageMode.MONGO == "mongo"
    assert StorageMode.LOCAL == "local"
    
    # Test conversion from string
    assert StorageMode("mongo") == StorageMode.MONGO
    assert StorageMode("local") == StorageMode.LOCAL


def test_env_settings():
    # Save original environment variables
    original_env = {
        "TELEGRAM_API_ID": os.environ.get("TELEGRAM_API_ID"),
        "TELEGRAM_API_HASH": os.environ.get("TELEGRAM_API_HASH"),
        "TELEGRAM_USER_ID": os.environ.get("TELEGRAM_USER_ID"),
        "TELETHON_SESSION_STR": os.environ.get("TELETHON_SESSION_STR"),
        "MONGO_CONN_STR": os.environ.get("MONGO_CONN_STR"),
        "MONGO_DB_NAME": os.environ.get("MONGO_DB_NAME"),
    }
    
    try:
        # Set test environment variables
        os.environ["TELEGRAM_API_ID"] = "12345"
        os.environ["TELEGRAM_API_HASH"] = "test_hash"
        os.environ["TELEGRAM_USER_ID"] = "67890"
        os.environ["TELETHON_SESSION_STR"] = "test_session"
        os.environ["MONGO_CONN_STR"] = "mongodb://localhost:27017"
        os.environ["MONGO_DB_NAME"] = "test_db"
        
        # Test loading from environment
        settings = TelegramDownloaderEnvSettings()
        
        assert settings.TELEGRAM_API_ID == 12345
        assert settings.TELEGRAM_API_HASH == "test_hash"
        assert settings.TELEGRAM_USER_ID == "67890"
        assert settings.TELETHON_SESSION_STR == "test_session"
        assert settings.MONGO_CONN_STR == "mongodb://localhost:27017"
        assert settings.MONGO_DB_NAME == "test_db"
        
        # Test default values
        assert settings.MONGO_MESSAGES_COLLECTION == "telegram_messages"
        assert settings.MONGO_CHATS_COLLECTION == "telegram_chats"
        assert settings.MONGO_USERS_COLLECTION == "telegram_users"
        assert settings.MONGO_HEARTBEATS_COLLECTION == "telegram_heartbeats"
        assert settings.MONGO_APP_DATA_COLLECTION == "telegram_downloader_app_data"
        
    finally:
        # Restore original environment variables
        for key, value in original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]


def test_size_thresholds():
    # Test defaults
    thresholds = SizeThresholds()
    assert thresholds.max_members_group == 1000
    assert thresholds.max_members_channel == 1000
    
    # Test custom values
    thresholds = SizeThresholds(max_members_group=2000, max_members_channel=3000)
    assert thresholds.max_members_group == 2000
    assert thresholds.max_members_channel == 3000


def test_chat_category_config():
    # Test defaults
    config = ChatCategoryConfig()
    assert config.enabled is False
    assert config.backdays is None
    assert config.limit is None
    assert config.download_attachments is False
    assert config.whitelist == []
    assert config.blacklist == []
    assert config.skip_big is True
    
    # Test custom values
    config = ChatCategoryConfig(
        enabled=True,
        backdays=30,
        limit=100,
        download_attachments=True,
        whitelist=["chat1", "chat2"],
        blacklist=["badchat"],
        skip_big=False,
    )
    
    assert config.enabled is True
    assert config.backdays == 30
    assert config.limit == 100
    assert config.download_attachments is True
    assert config.whitelist == ["chat1", "chat2"]
    assert config.blacklist == ["badchat"]
    assert config.skip_big is False


def test_telegram_downloader_config_defaults():
    config = TelegramDownloaderConfig()
    
    # Test default values
    assert config.storage_mode == StorageMode.MONGO
    assert isinstance(config.size_thresholds, SizeThresholds)
    assert isinstance(config.owned_groups, ChatCategoryConfig)
    assert isinstance(config.owned_channels, ChatCategoryConfig)
    assert isinstance(config.other_groups, ChatCategoryConfig)
    assert isinstance(config.other_channels, ChatCategoryConfig)
    assert isinstance(config.private_chats, ChatCategoryConfig)
    assert isinstance(config.bots, ChatCategoryConfig)


def test_telegram_downloader_config_from_yaml(sample_config_file, sample_config_dict):
    config = TelegramDownloaderConfig.from_yaml(sample_config_file)
    
    # Test main configuration
    assert config.storage_mode == StorageMode.MONGO
    
    # Test size thresholds
    assert config.size_thresholds.max_members_group == 1500
    assert config.size_thresholds.max_members_channel == 2000
    
    # Test owned groups config
    assert config.owned_groups.enabled is True
    assert config.owned_groups.backdays == 30
    assert config.owned_groups.limit == 1000
    assert config.owned_groups.whitelist == ["group1", "group2"]
    assert config.owned_groups.blacklist == ["badgroup"]
    assert config.owned_groups.download_attachments is True
    assert config.owned_groups.skip_big is False
    
    # Test owned channels config
    assert config.owned_channels.enabled is True
    assert config.owned_channels.backdays == 90
    assert config.owned_channels.limit == 500
    assert config.owned_channels.whitelist == []
    assert config.owned_channels.blacklist == []
    assert config.owned_channels.download_attachments is False
    assert config.owned_channels.skip_big is True
    
    # Test other categories with minimal specification
    assert config.other_groups.enabled is False
    assert config.other_channels.enabled is False
    assert config.private_chats.enabled is True
    assert config.private_chats.backdays == 365
    assert config.bots.enabled is False


def test_telegram_downloader_config_with_kwargs(sample_config_file):
    # Test override with kwargs
    config = TelegramDownloaderConfig.from_yaml(
        sample_config_file,
        storage_mode=StorageMode.LOCAL,
        owned_groups={"enabled": False},
        private_chats={"limit": 50},
    )
    
    assert config.storage_mode == StorageMode.LOCAL
    assert config.owned_groups.enabled is False  # Overridden
    assert config.private_chats.limit == 50  # Overridden
    assert config.private_chats.backdays == 365  # From file