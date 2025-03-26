import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest
from pymongo import MongoClient
from telethon.sessions import StringSession
from telethon.types import Dialog, Message

from telegram_downloader.config import (
    ChatCategoryConfig,
    StorageMode,
    TelegramDownloaderConfig,
)
from telegram_downloader.data_model import ChatData
from telegram_downloader.telegram_downloader import TelegramDownloader
from telegram_downloader.telethon_client_manager import TelethonClientManager


@pytest.fixture
def sample_config():
    """Create a sample configuration."""
    config = Mock(spec=TelegramDownloaderConfig)
    config.storage_mode = StorageMode.MONGO
    
    # Set up chat category configs
    owned_groups = Mock(spec=ChatCategoryConfig)
    owned_groups.enabled = True
    owned_groups.backdays = 30
    owned_groups.limit = 100
    owned_groups.skip_big = True
    
    owned_channels = Mock(spec=ChatCategoryConfig)
    owned_channels.enabled = True
    owned_channels.backdays = 60
    owned_channels.limit = 200
    owned_channels.skip_big = False
    
    other_groups = Mock(spec=ChatCategoryConfig)
    other_groups.enabled = False
    
    other_channels = Mock(spec=ChatCategoryConfig)
    other_channels.enabled = False
    
    private_chats = Mock(spec=ChatCategoryConfig)
    private_chats.enabled = True
    private_chats.backdays = 90
    private_chats.limit = 300
    private_chats.skip_big = False
    
    bots = Mock(spec=ChatCategoryConfig)
    bots.enabled = True
    bots.backdays = None
    bots.limit = 50
    bots.skip_big = False
    
    config.owned_groups = owned_groups
    config.owned_channels = owned_channels
    config.other_groups = other_groups
    config.other_channels = other_channels
    config.private_chats = private_chats
    config.bots = bots
    
    return config


@pytest.fixture
def mock_db():
    """Create a mock MongoDB database."""
    db = Mock()
    
    # Create mock collections
    messages_collection = Mock()
    chats_collection = Mock()
    users_collection = Mock()
    heartbeats_collection = Mock()
    app_data_collection = Mock()
    
    # Set up the collections
    db.__getitem__.side_effect = lambda key: {
        "telegram_messages": messages_collection,
        "telegram_chats": chats_collection,
        "telegram_users": users_collection,
        "telegram_heartbeats": heartbeats_collection,
        "telegram_downloader_app_data": app_data_collection,
    }.get(key, None)
    
    return db


@pytest.fixture
def mock_telethon_client():
    """Create a mock Telethon client."""
    client = AsyncMock()
    
    # Mock methods
    client.get_dialogs = AsyncMock()
    client.iter_messages = AsyncMock()
    
    return client


@pytest.fixture
def downloader(sample_config, mock_db, mock_telethon_client):
    """Create a TelegramDownloader instance with mocked dependencies."""
    # Mock TelegramDownloaderConfig.from_yaml
    with patch("telegram_downloader.telegram_downloader.TelegramDownloaderConfig") as mock_config_class:
        mock_config_class.from_yaml.return_value = sample_config
        
        # Mock TelegramDownloaderEnvSettings
        with patch("telegram_downloader.telegram_downloader.TelegramDownloaderEnvSettings") as mock_env_settings:
            env_settings = Mock()
            env_settings.TELEGRAM_API_ID = 12345
            env_settings.TELEGRAM_API_HASH = "test_hash"
            env_settings.TELEGRAM_USER_ID = "67890"
            env_settings.TELETHON_SESSION_STR = "test_session_string"
            env_settings.MONGO_CONN_STR = "mongodb://localhost:27017"
            env_settings.MONGO_DB_NAME = "test_db"
            env_settings.MONGO_MESSAGES_COLLECTION = "telegram_messages"
            env_settings.MONGO_CHATS_COLLECTION = "telegram_chats"
            env_settings.MONGO_USERS_COLLECTION = "telegram_users"
            env_settings.MONGO_HEARTBEATS_COLLECTION = "telegram_heartbeats"
            env_settings.MONGO_APP_DATA_COLLECTION = "telegram_downloader_app_data"
            mock_env_settings.return_value = env_settings
            
            # Create downloader with mocked dependencies
            downloader = TelegramDownloader(
                config_path=Path("config.yaml"),
                db=mock_db,
                telethon_client=mock_telethon_client
            )
            
            # Set up additional mocks
            downloader._chats = None
            downloader._chats_raw = None
            downloader._messages = None
            downloader._messages_raw = None
            downloader._chat_names = None
            
            yield downloader


def test_init(downloader, sample_config, mock_db, mock_telethon_client):
    """Test initialization of TelegramDownloader."""
    assert downloader.config == sample_config
    assert downloader._db == mock_db
    assert downloader._telethon_client == mock_telethon_client
    assert downloader.storage_mode == StorageMode.MONGO


def test_storage_mode_property(downloader, sample_config):
    """Test storage_mode property."""
    assert downloader.storage_mode == sample_config.storage_mode


@pytest.mark.asyncio
async def test_get_telethon_client_existing(downloader, mock_telethon_client):
    """Test get_telethon_client when client already exists."""
    client = await downloader.get_telethon_client()
    assert client == mock_telethon_client


@pytest.mark.asyncio
async def test_get_telethon_client_new_with_session_str(downloader):
    """Test get_telethon_client creating a new client with session string."""
    downloader._telethon_client = None
    
    with patch("telegram_downloader.telegram_downloader.TelegramClient") as mock_telegram_client:
        mock_client = AsyncMock()
        mock_telegram_client.return_value = mock_client
        
        client = await downloader.get_telethon_client()
        
        # Check client creation with session string
        mock_telegram_client.assert_called_once_with(
            StringSession(downloader.env.TELETHON_SESSION_STR),
            api_id=downloader.env.TELEGRAM_API_ID,
            api_hash=downloader.env.TELEGRAM_API_HASH
        )
        
        # Check client start was called
        mock_client.start.assert_called_once()
        
        # Check client was set and returned
        assert downloader._telethon_client == mock_client
        assert client == mock_client


@pytest.mark.asyncio
async def test_get_telethon_client_new_without_session_str(downloader):
    """Test get_telethon_client creating a new client without session string."""
    downloader._telethon_client = None
    downloader.env.TELETHON_SESSION_STR = None
    
    mock_client = AsyncMock()
    downloader._get_telethon_client = AsyncMock(return_value=mock_client)
    
    client = await downloader.get_telethon_client()
    
    # Check _get_telethon_client was called
    downloader._get_telethon_client.assert_called_once()
    
    # Check client was set and returned
    assert downloader._telethon_client == mock_client
    assert client == mock_client


@pytest.mark.asyncio
async def test_get_chat_list(downloader, mock_telethon_client):
    """Test _get_chat_list method."""
    # Mock dialogs
    dialogs = [Mock(spec=Dialog) for _ in range(3)]
    mock_telethon_client.get_dialogs.return_value = dialogs
    
    result = await downloader._get_chat_list()
    
    # Check that get_dialogs was called
    mock_telethon_client.get_dialogs.assert_called_once()
    
    # Check result
    assert result == dialogs


@pytest.mark.asyncio
async def test_get_chats_from_client(downloader, mock_telethon_client):
    """Test _get_chats_from_client method."""
    # Create mock dialogs
    mock_entity1 = Mock()
    mock_entity1.__class__.__name__ = "User"
    dialog1 = Mock(entity=mock_entity1, date=datetime.now(timezone.utc))
    
    mock_entity2 = Mock()
    mock_entity2.__class__.__name__ = "Channel"
    dialog2 = Mock(entity=mock_entity2, date=datetime.now(timezone.utc))
    
    mock_entity3 = Mock()
    mock_entity3.__class__.__name__ = "ChatForbidden"
    dialog3 = Mock(entity=mock_entity3, date=datetime.now(timezone.utc))
    
    mock_telethon_client.get_dialogs.return_value = [dialog1, dialog2, dialog3]
    
    # Mock the _filter_redundant_chats method
    filtered_chats = [ChatData(mock_entity1, dialog1.date), ChatData(mock_entity2, dialog2.date)]
    downloader._filter_redundant_chats = MagicMock(return_value=filtered_chats)
    
    result = await downloader._get_chats_from_client()
    
    # Check that get_dialogs was called
    mock_telethon_client.get_dialogs.assert_called_once()
    
    # Check that ChatForbidden was skipped and _filter_redundant_chats was called with the right chats
    downloader._filter_redundant_chats.assert_called_once()
    args = downloader._filter_redundant_chats.call_args[0][0]
    assert len(args) == 2
    assert all(isinstance(chat, ChatData) for chat in args)
    
    # Check result
    assert result == filtered_chats


def test_pick_chat_config(downloader):
    """Test _pick_chat_config method for different chat categories."""
    # Create mock chats for different categories
    owned_group = Mock(spec=ChatData)
    owned_group.entity_category = "group"
    owned_group.is_owned = True
    
    other_group = Mock(spec=ChatData)
    other_group.entity_category = "group"
    other_group.is_owned = False
    
    owned_channel = Mock(spec=ChatData)
    owned_channel.entity_category = "channel"
    owned_channel.is_owned = True
    
    other_channel = Mock(spec=ChatData)
    other_channel.entity_category = "channel"
    other_channel.is_owned = False
    
    bot = Mock(spec=ChatData)
    bot.entity_category = "bot"
    
    private_chat = Mock(spec=ChatData)
    private_chat.entity_category = "private chat"
    
    # Test each category
    assert downloader._pick_chat_config(owned_group) == downloader.config.owned_groups
    assert downloader._pick_chat_config(other_group) == downloader.config.other_groups
    assert downloader._pick_chat_config(owned_channel) == downloader.config.owned_channels
    assert downloader._pick_chat_config(other_channel) == downloader.config.other_channels
    assert downloader._pick_chat_config(bot) == downloader.config.bots
    assert downloader._pick_chat_config(private_chat) == downloader.config.private_chats
    
    # Test invalid category
    invalid_chat = Mock(spec=ChatData)
    invalid_chat.entity_category = "invalid"
    with pytest.raises(ValueError, match="Invalid chat category: invalid"):
        downloader._pick_chat_config(invalid_chat)


@pytest.mark.asyncio
async def test_get_chats_mongo_fresh(downloader):
    """Test _get_chats method with MongoDB storage and fresh chats."""
    # Mock has_fresh_chats_in_db to return True
    downloader._has_fresh_chats_in_db = MagicMock(return_value=True)
    
    # Mock load_chats_from_db
    mock_chats = [Mock(spec=ChatData) for _ in range(3)]
    downloader._load_chats_from_db = MagicMock(return_value=mock_chats)
    
    result = await downloader._get_chats()
    
    # Check methods were called correctly
    downloader._has_fresh_chats_in_db.assert_called_once()
    downloader._load_chats_from_db.assert_called_once()
    
    # Verify that _get_chats_from_client was not called
    assert not hasattr(downloader, "_get_chats_from_client.called")
    
    # Check result
    assert result == mock_chats


@pytest.mark.asyncio
async def test_get_chats_mongo_stale(downloader):
    """Test _get_chats method with MongoDB storage and stale chats."""
    # Mock has_fresh_chats_in_db to return False
    downloader._has_fresh_chats_in_db = MagicMock(return_value=False)
    
    # Mock get_chats_from_client
    mock_chats = [Mock(spec=ChatData) for _ in range(3)]
    downloader._get_chats_from_client = AsyncMock(return_value=mock_chats)
    
    # Mock save_chats_to_db
    downloader._save_chats_to_db = MagicMock()
    
    # Mock chat_refresh_timestamp setter
    with patch.object(TelegramDownloader, "chat_refresh_timestamp", new_callable=Mock()) as mock_setter:
        result = await downloader._get_chats()
        
        # Check methods were called correctly
        downloader._has_fresh_chats_in_db.assert_called_once()
        downloader._get_chats_from_client.assert_called_once()
        downloader._save_chats_to_db.assert_called_once_with(mock_chats)
        
        # Check timestamp was set
        assert mock_setter.called
        
        # Check result
        assert result == mock_chats


@pytest.mark.asyncio
async def test_get_chats_local_storage(downloader):
    """Test _get_chats method with local storage."""
    # Change storage mode to local
    downloader.config.storage_mode = StorageMode.LOCAL
    
    # Mock has_fresh_chats_on_disk to return True
    downloader._has_fresh_chats_on_disk = MagicMock(return_value=True)
    
    # Mock get_chats_from_disk
    mock_chats = [Mock(spec=ChatData) for _ in range(3)]
    downloader._get_chats_from_disk = MagicMock(return_value=mock_chats)
    
    result = await downloader._get_chats()
    
    # Check methods were called correctly
    downloader._has_fresh_chats_on_disk.assert_called_once()
    downloader._get_chats_from_disk.assert_called_once()
    
    # Check result
    assert result == mock_chats


@pytest.mark.asyncio
async def test_download_messages_disabled_config(downloader):
    """Test _download_messages with disabled config."""
    chat = Mock(spec=ChatData)
    chat.name = "Test Chat"
    
    chat_config = Mock(spec=ChatCategoryConfig)
    chat_config.enabled = False
    
    result = await downloader._download_messages(chat, chat_config)
    
    # Check that no further methods were called
    assert result == []


@pytest.mark.asyncio
async def test_download_messages_old_chat(downloader):
    """Test _download_messages with chat older than backdays."""
    chat = Mock(spec=ChatData)
    chat.name = "Test Chat"
    chat.last_message_date = datetime.now(timezone.utc) - timedelta(days=100)
    
    chat_config = Mock(spec=ChatCategoryConfig)
    chat_config.enabled = True
    chat_config.backdays = 30  # Only get messages from the last 30 days
    
    # Mock _save_chat_to_db
    downloader._save_chat_to_db = MagicMock()
    
    result = await downloader._download_messages(chat, chat_config)
    
    # Check that chat was marked as finished and saved
    assert chat.finished_downloading is True
    downloader._save_chat_to_db.assert_called_once_with(chat)
    
    # Check result
    assert result == []


@pytest.mark.asyncio
async def test_download_messages_big_chat(downloader):
    """Test _download_messages with big chat and skip_big=True."""
    chat = Mock(spec=ChatData)
    chat.name = "Test Chat"
    chat.is_big = True
    chat.last_message_date = datetime.now(timezone.utc)
    
    chat_config = Mock(spec=ChatCategoryConfig)
    chat_config.enabled = True
    chat_config.backdays = None
    chat_config.skip_big = True
    
    result = await downloader._download_messages(chat, chat_config)
    
    # Check result
    assert result == []


@pytest.mark.asyncio
async def test_download_messages_no_existing_finished(downloader):
    """Test _download_messages with no existing messages and chat marked as finished."""
    chat = Mock(spec=ChatData)
    chat.name = "Test Chat"
    chat.id = 12345
    chat.is_big = False
    chat.last_message_date = datetime.now(timezone.utc)
    chat.finished_downloading = True
    
    chat_config = Mock(spec=ChatCategoryConfig)
    chat_config.enabled = True
    chat_config.backdays = None
    chat_config.skip_big = False
    
    # Mock _get_chat_message_range to return None, None (no existing messages)
    downloader._get_chat_message_range = MagicMock(return_value=(None, None))
    
    result = await downloader._download_messages(chat, chat_config)
    
    # Check that we skipped the chat due to it being marked as finished
    assert result == []


@pytest.mark.asyncio
async def test_download_messages_no_existing_not_finished(downloader):
    """Test _download_messages with no existing messages and chat not marked as finished."""
    chat = Mock(spec=ChatData)
    chat.name = "Test Chat"
    chat.id = 12345
    chat.is_big = False
    chat.last_message_date = datetime.now(timezone.utc)
    chat.finished_downloading = False
    
    chat_config = Mock(spec=ChatCategoryConfig)
    chat_config.enabled = True
    chat_config.backdays = None
    chat_config.skip_big = False
    
    # Mock _get_chat_message_range to return None, None (no existing messages)
    downloader._get_chat_message_range = MagicMock(return_value=(None, None))
    
    # Mock _download_all_messages
    mock_messages = [Mock(spec=Message) for _ in range(3)]
    downloader._download_all_messages = AsyncMock(return_value=mock_messages)
    
    # Mock _save_chat_to_db
    downloader._save_chat_to_db = MagicMock()
    
    result = await downloader._download_messages(chat, chat_config)
    
    # Check methods were called correctly
    downloader._download_all_messages.assert_called_once_with(chat, chat_config)
    
    # Check that chat was marked as finished and saved
    assert chat.finished_downloading is True
    downloader._save_chat_to_db.assert_called_once_with(chat)
    
    # Check result
    assert result == mock_messages


@pytest.mark.asyncio
async def test_download_messages_existing_messages(downloader):
    """Test _download_messages with existing messages."""
    chat = Mock(spec=ChatData)
    chat.name = "Test Chat"
    chat.id = 12345
    chat.is_big = False
    chat.last_message_date = datetime.now(timezone.utc)
    chat.finished_downloading = False
    
    chat_config = Mock(spec=ChatCategoryConfig)
    chat_config.enabled = True
    chat_config.backdays = None
    chat_config.skip_big = False
    
    # Mock _get_chat_message_range to return existing message range
    min_time = datetime.now(timezone.utc) - timedelta(days=10)
    max_time = datetime.now(timezone.utc) - timedelta(days=1)
    downloader._get_chat_message_range = MagicMock(return_value=(min_time, max_time))
    
    # Mock _download_messages_excluding_range
    mock_messages = [Mock(spec=Message) for _ in range(3)]
    downloader._download_messages_excluding_range = AsyncMock(return_value=mock_messages)
    
    # Mock _save_chat_to_db
    downloader._save_chat_to_db = MagicMock()
    
    result = await downloader._download_messages(chat, chat_config)
    
    # Check methods were called correctly
    downloader._download_messages_excluding_range.assert_called_once_with(
        chat, chat_config, min_time, max_time, ignore_finished=False
    )
    
    # Check that chat was marked as finished and saved
    assert chat.finished_downloading is True
    downloader._save_chat_to_db.assert_called_once_with(chat)
    
    # Check result
    assert result == mock_messages


@pytest.mark.asyncio
async def test_download_all_messages(downloader, mock_telethon_client):
    """Test _download_all_messages method."""
    chat = Mock(spec=ChatData)
    chat.entity = Mock()
    
    chat_config = Mock(spec=ChatCategoryConfig)
    chat_config.backdays = 30
    chat_config.limit = 100
    
    # Mock _load_messages
    mock_messages = [Mock(spec=Message) for _ in range(3)]
    downloader._load_messages = AsyncMock(return_value=mock_messages)
    
    result = await downloader._download_all_messages(chat, chat_config)
    
    # Check that _load_messages was called with correct parameters
    downloader._load_messages.assert_called_once()
    args, kwargs = downloader._load_messages.call_args
    assert args[0] == chat
    assert 'offset_date' in kwargs
    assert 'reverse' in kwargs
    assert kwargs['reverse'] is True
    assert kwargs['limit'] == 100
    
    # Check result
    assert result == mock_messages


@pytest.mark.asyncio
async def test_download_messages_excluding_range(downloader):
    """Test _download_messages_excluding_range method."""
    chat = Mock(spec=ChatData)
    chat.entity = Mock()
    chat.finished_downloading = False
    
    chat_config = Mock(spec=ChatCategoryConfig)
    chat_config.limit = 100
    
    min_timestamp = datetime.now(timezone.utc) - timedelta(days=10)
    max_timestamp = datetime.now(timezone.utc) - timedelta(days=1)
    
    # Mock _load_messages for newer and older messages
    newer_messages = [Mock(spec=Message) for _ in range(2)]
    older_messages = [Mock(spec=Message) for _ in range(3)]
    
    downloader._load_messages = AsyncMock(side_effect=[newer_messages, older_messages])
    
    result = await downloader._download_messages_excluding_range(
        chat, chat_config, min_timestamp, max_timestamp
    )
    
    # Check _load_messages calls
    assert downloader._load_messages.call_count == 2
    
    # First call: newer messages
    first_call_args, first_call_kwargs = downloader._load_messages.call_args_list[0]
    assert first_call_args[0] == chat
    assert first_call_kwargs['offset_date'] == max_timestamp
    assert first_call_kwargs['reverse'] is True
    
    # Second call: older messages
    second_call_args, second_call_kwargs = downloader._load_messages.call_args_list[1]
    assert second_call_args[0] == chat
    assert second_call_kwargs['offset_date'] == min_timestamp
    assert second_call_kwargs['reverse'] is False
    assert second_call_kwargs['limit'] == 100
    
    # Check result combines both sets of messages
    assert result == newer_messages + older_messages


@pytest.mark.asyncio
async def test_download_messages_excluding_range_finished(downloader):
    """Test _download_messages_excluding_range with finished chat."""
    chat = Mock(spec=ChatData)
    chat.entity = Mock()
    chat.finished_downloading = True
    
    chat_config = Mock(spec=ChatCategoryConfig)
    chat_config.limit = 100
    
    min_timestamp = datetime.now(timezone.utc) - timedelta(days=10)
    max_timestamp = datetime.now(timezone.utc) - timedelta(days=1)
    
    # Mock _load_messages for newer messages only
    newer_messages = [Mock(spec=Message) for _ in range(2)]
    downloader._load_messages = AsyncMock(return_value=newer_messages)
    
    result = await downloader._download_messages_excluding_range(
        chat, chat_config, min_timestamp, max_timestamp
    )
    
    # Check _load_messages was only called once (for newer messages)
    downloader._load_messages.assert_called_once()
    
    # Check result only includes newer messages
    assert result == newer_messages


@pytest.mark.asyncio
async def test_load_messages(downloader, mock_telethon_client):
    """Test _load_messages method."""
    chat = Mock(spec=ChatData)
    chat.name = "Test Chat"
    chat.entity = Mock()
    
    offset_date = datetime.now(timezone.utc) - timedelta(days=5)
    reverse = True
    limit = 50
    
    # Mock the async iterator for client.iter_messages
    message1 = Mock(spec=Message)
    message2 = Mock(spec=Message)
    
    mock_telethon_client.iter_messages = MagicMock()
    mock_telethon_client.iter_messages.return_value.__aiter__.return_value = [message1, message2]
    
    result = await downloader._load_messages(
        chat, offset_date=offset_date, reverse=reverse, limit=limit
    )
    
    # Check that client.iter_messages was called with correct parameters
    mock_telethon_client.iter_messages.assert_called_once_with(
        chat.entity, offset_date=offset_date, reverse=reverse, limit=limit
    )
    
    # Check result
    assert result == [message1, message2]


@pytest.mark.asyncio
async def test_load_messages_error(downloader, mock_telethon_client):
    """Test _load_messages method with error."""
    chat = Mock(spec=ChatData)
    chat.name = "Test Chat"
    chat.entity = Mock()
    
    # Make the iterator raise an exception
    mock_telethon_client.iter_messages = MagicMock()
    mock_telethon_client.iter_messages.return_value.__aiter__.return_value = AsyncMock(
        side_effect=Exception("Test error")
    )
    
    result = await downloader._load_messages(chat)
    
    # Check result is empty on error
    assert result == []


def test_filter_redundant_chats(downloader):
    """Test _filter_redundant_chats method."""
    # Create mock entities
    chat1 = Mock(spec=ChatData)
    chat1.id = 1
    chat1.name = "Chat 1"
    chat1.entity = Mock()
    chat1.entity.migrated_to = None
    
    chat2 = Mock(spec=ChatData)
    chat2.id = 2
    chat2.name = "Chat 2"
    chat2.entity = Mock()
    chat2.entity.migrated_to = Mock()
    chat2.entity.migrated_to.channel_id = 3
    
    chat3 = Mock(spec=ChatData)
    chat3.id = 3
    chat3.name = "Channel 3 (migrated from Chat 2)"
    chat3.entity = Mock()
    chat3.entity.migrated_to = None
    
    chat4 = Mock(spec=ChatData)
    chat4.id = 4
    chat4.name = "Chat 4"
    chat4.entity = Mock()
    chat4.entity.migrated_to = Mock()
    chat4.entity.migrated_to.channel_id = 5
    
    # The channel that chat4 was migrated to doesn't exist in our list
    
    chats = [chat1, chat2, chat3, chat4]
    
    result = downloader._filter_redundant_chats(chats)
    
    # Check that chat2 was filtered out (migrated to chat3) but chat4 remains
    assert len(result) == 3
    assert chat1 in result
    assert chat2 not in result  # Filtered out because chat3 exists
    assert chat3 in result
    assert chat4 in result  # Kept because its migration target doesn't exist