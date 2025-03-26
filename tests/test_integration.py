import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from pymongo import MongoClient

from telegram_downloader.config import StorageMode
from telegram_downloader.data_model import ChatData
from telegram_downloader.telegram_downloader import TelegramDownloader
from telegram_downloader.telethon_client_manager import TelethonClientManager


@pytest.fixture
def mock_env_variables():
    """Set up environment variables for testing."""
    # Save original environment variables
    original_env = {}
    env_vars = [
        "TELEGRAM_API_ID", 
        "TELEGRAM_API_HASH", 
        "TELEGRAM_USER_ID",
        "TELETHON_SESSION_STR",
        "MONGO_CONN_STR",
        "MONGO_DB_NAME"
    ]
    
    for var in env_vars:
        original_env[var] = os.environ.get(var)
    
    # Set test environment variables
    os.environ["TELEGRAM_API_ID"] = "12345"
    os.environ["TELEGRAM_API_HASH"] = "test_hash"
    os.environ["TELEGRAM_USER_ID"] = "67890"
    os.environ["TELETHON_SESSION_STR"] = "test_session_string"
    os.environ["MONGO_CONN_STR"] = "mongodb://localhost:27017"
    os.environ["MONGO_DB_NAME"] = "test_db"
    
    yield
    
    # Restore original environment variables
    for var in env_vars:
        if original_env[var] is not None:
            os.environ[var] = original_env[var]
        elif var in os.environ:
            del os.environ[var]


@pytest.fixture
def sample_config_yaml():
    """Create a sample configuration YAML file."""
    config_content = """
storage_mode: mongo

size_thresholds:
  max_members_group: 1000
  max_members_channel: 1000

owned_groups:
  enabled: true
  backdays: 30
  limit: 100
  whitelist: []
  blacklist: []
  download_attachments: false
  skip_big: true

owned_channels:
  enabled: true
  backdays: 30
  limit: 100
  whitelist: []
  blacklist: []
  download_attachments: false
  skip_big: true

other_groups:
  enabled: false
  backdays: 30
  limit: 100
  whitelist: []
  blacklist: []
  download_attachments: false
  skip_big: true

other_channels:
  enabled: false
  backdays: 30
  limit: 100
  whitelist: []
  blacklist: []
  download_attachments: false
  skip_big: true

private_chats:
  enabled: true
  backdays: 60
  limit: 200
  whitelist: []
  blacklist: []
  download_attachments: false
  skip_big: false

bots:
  enabled: true
  backdays: 90
  limit: 50
  whitelist: []
  blacklist: []
  download_attachments: false
  skip_big: false
"""
    # Create temporary config file
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
        temp_file.write(config_content)
        config_path = Path(temp_file.name)
    
    yield config_path
    
    # Clean up
    if config_path.exists():
        config_path.unlink()


@pytest.fixture
def mock_mongo_client():
    """Mock MongoDB client with collections."""
    with patch("telegram_downloader.telegram_downloader.MongoClient") as mock_client_class:
        # Create mock client and database
        mock_client = Mock()
        mock_db = Mock()
        mock_client.return_value = mock_client
        mock_client.__getitem__.return_value = mock_db
        
        # Create mock collections
        messages_collection = Mock()
        chats_collection = Mock()
        users_collection = Mock()
        heartbeats_collection = Mock()
        app_data_collection = Mock()
        
        # Configure find_one and find methods
        app_data_collection.find_one.return_value = None  # No refresh timestamp initially
        chats_collection.find.return_value = []  # No chats initially
        
        # Set up the collections dict for database
        mock_db.__getitem__.side_effect = lambda key: {
            "telegram_messages": messages_collection,
            "telegram_chats": chats_collection,
            "telegram_users": users_collection,
            "telegram_heartbeats": heartbeats_collection,
            "telegram_downloader_app_data": app_data_collection,
        }.get(key, Mock())
        
        # Make database.list_collection_names() return empty list
        mock_db.list_collection_names.return_value = []
        
        # Make database.create_collection() return mock collections
        mock_db.create_collection.side_effect = lambda name: Mock()
        
        # Make client.list_database_names() return empty list
        mock_client.list_database_names.return_value = []
        
        yield mock_client


@pytest.fixture
def mock_telethon_client():
    """Mock Telethon client with dialogs and messages."""
    with patch("telegram_downloader.telegram_downloader.TelegramClient") as mock_client_class:
        # Create mock client
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        
        # Mock is_user_authorized to return True
        mock_client.is_user_authorized.return_value = True
        
        # Create mock dialogs
        user = Mock()
        user.id = 123
        user.first_name = "Test"
        user.last_name = "User"
        user.username = "testuser"
        user.bot = False
        user.__class__.__name__ = "User"
        
        bot = Mock()
        bot.id = 456
        bot.first_name = "Test"
        bot.last_name = "Bot"
        bot.username = "testbot"
        bot.bot = True
        bot.__class__.__name__ = "User"
        
        channel = Mock()
        channel.id = 789
        channel.title = "Test Channel"
        channel.username = "testchannel"
        channel.participants_count = 5000
        channel.megagroup = False
        channel.creator = True
        channel.__class__.__name__ = "Channel"
        
        group = Mock()
        group.id = 321
        group.title = "Test Group"
        group.username = "testgroup"
        group.participants_count = 200
        group.megagroup = True
        group.creator = False
        group.__class__.__name__ = "Channel"
        
        # Create dialogs
        dialog1 = Mock()
        dialog1.entity = user
        dialog1.date = datetime.now(timezone.utc)
        dialog1.name = "Test User"
        dialog1.id = user.id
        
        dialog2 = Mock()
        dialog2.entity = bot
        dialog2.date = datetime.now(timezone.utc)
        dialog2.name = "Test Bot"
        dialog2.id = bot.id
        
        dialog3 = Mock()
        dialog3.entity = channel
        dialog3.date = datetime.now(timezone.utc)
        dialog3.name = "Test Channel"
        dialog3.id = channel.id
        
        dialog4 = Mock()
        dialog4.entity = group
        dialog4.date = datetime.now(timezone.utc)
        dialog4.name = "Test Group"
        dialog4.id = group.id
        
        mock_client.get_dialogs.return_value = [dialog1, dialog2, dialog3, dialog4]
        
        # Mock iter_messages
        message1 = Mock()
        message1.id = 1001
        message1.peer_id = Mock()
        message1.peer_id.user_id = user.id
        message1.peer_id._ = "PeerUser"
        message1.date = datetime.now(timezone.utc) - timedelta(days=1)
        message1.text = "Test message 1"
        message1.to_dict.return_value = {
            "id": message1.id,
            "peer_id": {"_": "PeerUser", "user_id": user.id},
            "date": message1.date,
            "text": message1.text
        }
        
        message2 = Mock()
        message2.id = 1002
        message2.peer_id = Mock()
        message2.peer_id.channel_id = channel.id
        message2.peer_id._ = "PeerChannel"
        message2.date = datetime.now(timezone.utc) - timedelta(days=2)
        message2.text = "Test message 2"
        message2.to_dict.return_value = {
            "id": message2.id,
            "peer_id": {"_": "PeerChannel", "channel_id": channel.id},
            "date": message2.date,
            "text": message2.text
        }
        
        # Set up the async iterator for iter_messages
        async def mock_iter_messages(*args, **kwargs):
            entity = args[0]
            if hasattr(entity, 'id') and entity.id == user.id:
                yield message1
            elif hasattr(entity, 'id') and entity.id == channel.id:
                yield message2
        
        mock_client.iter_messages = MagicMock()
        mock_client.iter_messages.return_value.__aiter__ = MagicMock(side_effect=lambda: mock_iter_messages())
        
        yield mock_client


@pytest.mark.asyncio
async def test_integration_main_flow(mock_env_variables, sample_config_yaml, mock_mongo_client, mock_telethon_client):
    """Test the main flow of TelegramDownloader."""
    # Create TelegramDownloader instance
    downloader = TelegramDownloader(config_path=sample_config_yaml)
    
    # Mock get_telethon_client to return our mock client
    downloader._telethon_client = mock_telethon_client
    
    # Run the main method
    await downloader.main(chat_sample_size=2)
    
    # Check that the methods were called in the correct order
    mock_telethon_client.get_dialogs.assert_called()
    
    # Check that messages were saved to the database
    messages_collection = downloader.messages_collection
    messages_collection.insert_many.assert_called()
    
    # Check that duplicate message cleanup was run
    assert downloader.delete_duplicate_messages.called


@pytest.mark.asyncio
async def test_integration_demo_chat_stats(mock_env_variables, sample_config_yaml, mock_mongo_client, mock_telethon_client):
    """Test the demo_chat_stats method."""
    # Create TelegramDownloader instance
    downloader = TelegramDownloader(config_path=sample_config_yaml)
    
    # Mock get_telethon_client to return our mock client
    downloader._telethon_client = mock_telethon_client
    
    # Mock calculate_stats and get_random_chat to focus on testing flow
    downloader.calculate_stats = MagicMock(return_value={
        "groups": 10,
        "channels": 5,
        "bots": 3,
        "private chats": 20,
        "recent_groups": 5,
        "owned_channels": 2
    })
    
    downloader.get_random_chat = MagicMock(return_value={
        "entity": Mock(
            title="Random Chat",
            username="randomchat"
        )
    })
    
    # Run the demo_chat_stats method
    await downloader.demo_chat_stats()
    
    # Check that the methods were called
    mock_telethon_client.get_dialogs.assert_called()
    downloader.calculate_stats.assert_called_once()
    assert downloader.get_random_chat.call_count >= 1