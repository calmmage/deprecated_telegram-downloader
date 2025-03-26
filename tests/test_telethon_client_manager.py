import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from telegram_downloader.config import StorageMode
from telegram_downloader.telethon_client_manager import (
    TelethonClientManager,
    TelethonClientManagerEnvSettings,
)


@pytest.fixture
def mock_env_settings():
    with patch.dict(os.environ, {
        "TELEGRAM_API_ID": "12345",
        "TELEGRAM_API_HASH": "test_hash",
        "SESSIONS_DIR": "/tmp/sessions",
    }):
        yield


@pytest.fixture
def telethon_client_manager():
    # Mock the TelethonClientManagerEnvSettings initialization
    with patch("telegram_downloader.telethon_client_manager.TelethonClientManagerEnvSettings") as mock_settings:
        settings_instance = Mock()
        settings_instance.TELEGRAM_API_ID = 12345
        settings_instance.TELEGRAM_API_HASH = "test_hash"
        settings_instance.SESSIONS_DIR = Path("/tmp/sessions")
        settings_instance.MONGO_CONN_STR = None
        mock_settings.return_value = settings_instance
        
        # Create the client manager
        manager = TelethonClientManager(storage_mode=StorageMode.LOCAL)
        
        # Set up additional attributes for testing
        manager.clients = {}
        
        yield manager


def test_env_settings(mock_env_settings):
    """Test that environment settings are loaded correctly."""
    settings = TelethonClientManagerEnvSettings()
    
    assert settings.TELEGRAM_API_ID == 12345
    assert settings.TELEGRAM_API_HASH == "test_hash"
    assert settings.SESSIONS_DIR == Path("/tmp/sessions")


def test_init(telethon_client_manager):
    """Test that the client manager initializes correctly."""
    manager = telethon_client_manager
    
    assert manager.storage_mode == StorageMode.LOCAL
    assert manager.api_id == 12345
    assert manager.api_hash == "test_hash"
    assert manager.sessions_dir == Path("/tmp/sessions")
    assert manager.clients == {}


@pytest.mark.asyncio
async def test_get_telethon_client_local_storage(telethon_client_manager):
    """Test that get_telethon_client routes correctly for local storage."""
    manager = telethon_client_manager
    user_id = 67890
    
    # Mock the _get_telethon_client_from_disk method
    manager._get_telethon_client_from_disk = AsyncMock()
    
    await manager.get_telethon_client(user_id)
    
    # Check that the correct method was called
    manager._get_telethon_client_from_disk.assert_called_once_with(user_id)


@pytest.mark.asyncio
async def test_get_telethon_client_mongo_storage(telethon_client_manager):
    """Test that get_telethon_client routes correctly for mongo storage."""
    manager = telethon_client_manager
    manager.storage_mode = StorageMode.MONGO
    user_id = 67890
    
    # Mock the _get_telethon_client_from_database method
    manager._get_telethon_client_from_database = AsyncMock()
    
    await manager.get_telethon_client(user_id)
    
    # Check that the correct method was called
    manager._get_telethon_client_from_database.assert_called_once_with(user_id)


@pytest.mark.asyncio
async def test_get_telethon_client_invalid_storage(telethon_client_manager):
    """Test that get_telethon_client raises an error for invalid storage mode."""
    manager = telethon_client_manager
    manager.storage_mode = "invalid"  # Invalid storage mode
    user_id = 67890
    
    with pytest.raises(ValueError, match="Invalid storage mode: invalid"):
        await manager.get_telethon_client(user_id)


@pytest.mark.asyncio
async def test_check_if_conn_is_present_on_disk(telethon_client_manager):
    """Test checking if connection exists on disk."""
    manager = telethon_client_manager
    user_id = 67890
    session_file = manager.sessions_dir / f"user_{user_id}.session"
    
    # Test when file doesn't exist
    with patch.object(Path, "exists", return_value=False):
        assert not manager._check_if_conn_is_present_on_disk(user_id)
    
    # Test when file exists
    with patch.object(Path, "exists", return_value=True):
        assert manager._check_if_conn_is_present_on_disk(user_id)


@pytest.mark.asyncio
async def test_load_conn_from_disk_success(telethon_client_manager):
    """Test loading connection from disk successfully."""
    manager = telethon_client_manager
    user_id = 67890
    session_file = manager.sessions_dir / f"user_{user_id}.session"
    
    # Mock file existence check
    with patch.object(Path, "exists", return_value=True):
        # Mock TelegramClient
        with patch("telegram_downloader.telethon_client_manager.TelegramClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Mock successful connection
            mock_client.connect = AsyncMock()
            mock_client.is_user_authorized = AsyncMock(return_value=True)
            
            result = await manager._load_conn_from_disk(user_id)
            
            # Check that the client was created with correct parameters
            mock_client_class.assert_called_once_with(
                str(manager.sessions_dir / f"user_{user_id}"),
                manager.api_id,
                manager.api_hash
            )
            
            # Check that the client was connected and authorized
            mock_client.connect.assert_called_once()
            mock_client.is_user_authorized.assert_called_once()
            
            # Check that the client was added to the manager's clients dictionary
            assert manager.clients[user_id] == mock_client
            
            # Check that the correct client was returned
            assert result == mock_client


@pytest.mark.asyncio
async def test_load_conn_from_disk_file_not_found(telethon_client_manager):
    """Test loading connection from disk when session file doesn't exist."""
    manager = telethon_client_manager
    user_id = 67890
    
    # Mock file existence check
    with patch.object(Path, "exists", return_value=False):
        with pytest.raises(Exception, match="No session file found"):
            await manager._load_conn_from_disk(user_id)


@pytest.mark.asyncio
async def test_load_conn_from_disk_not_authorized(telethon_client_manager):
    """Test loading connection from disk when client is not authorized."""
    manager = telethon_client_manager
    user_id = 67890
    
    # Mock file existence check
    with patch.object(Path, "exists", return_value=True):
        # Mock TelegramClient
        with patch("telegram_downloader.telethon_client_manager.TelegramClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Mock successful connection but failed authorization
            mock_client.connect = AsyncMock()
            mock_client.is_user_authorized = AsyncMock(return_value=False)
            
            with pytest.raises(Exception, match="Session exists but not authorized"):
                await manager._load_conn_from_disk(user_id)


@pytest.mark.asyncio
async def test_get_telethon_client_from_disk_existing(telethon_client_manager):
    """Test getting client from disk when session exists."""
    manager = telethon_client_manager
    user_id = 67890
    
    # Mock checking if connection exists
    manager._check_if_conn_is_present_on_disk = MagicMock(return_value=True)
    
    # Mock loading connection from disk
    mock_client = AsyncMock()
    manager._load_conn_from_disk = AsyncMock(return_value=mock_client)
    
    result = await manager._get_telethon_client_from_disk(user_id)
    
    # Check that the correct methods were called
    manager._check_if_conn_is_present_on_disk.assert_called_once_with(user_id)
    manager._load_conn_from_disk.assert_called_once_with(user_id)
    
    # Check that the correct client was returned
    assert result == mock_client


@pytest.mark.asyncio
async def test_get_telethon_client_from_disk_new(telethon_client_manager):
    """Test getting client from disk when session doesn't exist."""
    manager = telethon_client_manager
    user_id = 67890
    
    # Mock checking if connection exists
    manager._check_if_conn_is_present_on_disk = MagicMock(return_value=False)
    
    # Mock creating new connection
    mock_client = AsyncMock()
    manager._create_new_telethon_client_and_save_to_disk = AsyncMock(return_value=mock_client)
    
    result = await manager._get_telethon_client_from_disk(user_id)
    
    # Check that the correct methods were called
    manager._check_if_conn_is_present_on_disk.assert_called_once_with(user_id)
    manager._create_new_telethon_client_and_save_to_disk.assert_called_once_with(user_id)
    
    # Check that the correct client was returned
    assert result == mock_client


@pytest.mark.asyncio
async def test_database_methods_not_implemented(telethon_client_manager):
    """Test that database methods raise NotImplementedError."""
    manager = telethon_client_manager
    user_id = 67890
    
    with pytest.raises(NotImplementedError, match="Database check not implemented yet"):
        await manager._check_if_conn_is_present_in_db(user_id)
    
    with pytest.raises(NotImplementedError, match="Database loading not implemented yet"):
        await manager._load_conn_from_db(user_id)
    
    with pytest.raises(NotImplementedError, match="Database storage not implemented yet"):
        await manager._create_new_telethon_client_and_save_to_db(user_id)