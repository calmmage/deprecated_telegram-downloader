import json
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from telegram_downloader.data_model import ChatData

# Create mock entities for testing
@pytest.fixture
def mock_user():
    user = Mock()
    user.__class__.__name__ = "User"
    user.id = 1234
    user.first_name = "Test"
    user.last_name = "User"
    user.username = "testuser"
    user.bot = False
    return user

@pytest.fixture
def mock_bot():
    bot = Mock()
    bot.__class__.__name__ = "User"
    bot.id = 5678
    bot.first_name = "Bot"
    bot.last_name = None
    bot.username = "testbot"
    bot.bot = True
    return bot

@pytest.fixture
def mock_channel():
    channel = Mock()
    channel.__class__.__name__ = "Channel"
    channel.id = 9012
    channel.title = "Test Channel"
    channel.username = "testchannel"
    channel.participants_count = 5000
    channel.megagroup = False
    channel.creator = True
    channel.to_dict = lambda: {"_": "Channel", "id": channel.id, "title": channel.title, 
                              "username": channel.username, "participants_count": channel.participants_count,
                              "megagroup": channel.megagroup, "creator": channel.creator}
    return channel

@pytest.fixture
def mock_group():
    group = Mock()
    group.__class__.__name__ = "Channel"
    group.id = 3456
    group.title = "Test Group"
    group.username = "testgroup"
    group.participants_count = 200
    group.megagroup = True
    group.creator = False
    group.to_dict = lambda: {"_": "Channel", "id": group.id, "title": group.title, 
                            "username": group.username, "participants_count": group.participants_count,
                            "megagroup": group.megagroup, "creator": group.creator}
    return group

@pytest.fixture
def mock_small_chat():
    chat = Mock()
    chat.__class__.__name__ = "Chat"
    chat.id = 7890
    chat.title = "Small Chat"
    chat.username = None
    chat.participants_count = 5
    chat.creator = False
    chat.to_dict = lambda: {"_": "Chat", "id": chat.id, "title": chat.title, 
                           "participants_count": chat.participants_count, "creator": chat.creator}
    return chat


def test_chat_data_init(mock_user, mock_channel):
    # Test with user entity
    now = datetime.now(timezone.utc)
    chat_data = ChatData(mock_user, now)
    
    assert chat_data.entity == mock_user
    assert chat_data.last_message_date == now
    assert chat_data.finished_downloading is False
    assert chat_data.id == mock_user.id
    
    # Test with channel entity
    chat_data = ChatData(mock_channel, now, finished_downloading=True)
    
    assert chat_data.entity == mock_channel
    assert chat_data.last_message_date == now
    assert chat_data.finished_downloading is True
    assert chat_data.id == mock_channel.id


def test_entity_category(mock_user, mock_bot, mock_channel, mock_group, mock_small_chat):
    # Test user (private chat)
    chat_data = ChatData(mock_user, datetime.now(timezone.utc))
    assert chat_data.entity_category == "private chat"
    
    # Test bot
    chat_data = ChatData(mock_bot, datetime.now(timezone.utc))
    assert chat_data.entity_category == "bot"
    
    # Test channel
    chat_data = ChatData(mock_channel, datetime.now(timezone.utc))
    assert chat_data.entity_category == "channel"
    
    # Test group (megagroup channel)
    chat_data = ChatData(mock_group, datetime.now(timezone.utc))
    assert chat_data.entity_category == "group"
    
    # Test small chat
    chat_data = ChatData(mock_small_chat, datetime.now(timezone.utc))
    assert chat_data.entity_category == "group"


def test_is_recent(mock_user):
    # Test with recent date
    now = datetime.now(timezone.utc)
    chat_data = ChatData(mock_user, now)
    assert chat_data.is_recent is True
    
    # Test with old date
    old_date = now - timedelta(days=60)
    chat_data = ChatData(mock_user, old_date)
    assert chat_data.is_recent is False
    
    # Test with custom threshold
    chat_data = ChatData(mock_user, old_date)
    assert chat_data.get_is_recent(timedelta(days=90)) is True
    assert chat_data.get_is_recent(timedelta(days=30)) is False


def test_is_owned(mock_channel, mock_group):
    # Test with owned channel
    chat_data = ChatData(mock_channel, datetime.now(timezone.utc))
    assert chat_data.is_owned is True
    
    # Test with not owned group
    chat_data = ChatData(mock_group, datetime.now(timezone.utc))
    assert chat_data.is_owned is False


def test_is_big(mock_channel, mock_group, mock_small_chat):
    # Test with big channel
    chat_data = ChatData(mock_channel, datetime.now(timezone.utc))
    assert chat_data.is_big is True
    
    # Test with small group
    chat_data = ChatData(mock_group, datetime.now(timezone.utc))
    assert chat_data.is_big is False
    
    # Test with very small chat
    chat_data = ChatData(mock_small_chat, datetime.now(timezone.utc))
    assert chat_data.is_big is False
    
    # Test with custom threshold
    chat_data = ChatData(mock_group, datetime.now(timezone.utc))
    assert chat_data.get_is_big(100) is True
    assert chat_data.get_is_big(500) is False


def test_name_formatting(mock_user, mock_channel, mock_small_chat):
    # Test user name formatting
    chat_data = ChatData(mock_user, datetime.now(timezone.utc))
    assert chat_data.name == "Test User @testuser [1234]"
    
    # Test channel name formatting
    chat_data = ChatData(mock_channel, datetime.now(timezone.utc))
    assert chat_data.name == "Test Channel @testchannel [9012]"
    
    # Test chat without username formatting
    chat_data = ChatData(mock_small_chat, datetime.now(timezone.utc))
    assert chat_data.name == "Small Chat [7890]"


def test_to_dict(mock_channel):
    now = datetime.now(timezone.utc)
    chat_data = ChatData(mock_channel, now)
    
    data_dict = chat_data.to_dict()
    
    assert data_dict["id"] == mock_channel.id
    assert data_dict["entity"] == mock_channel.to_dict()
    assert data_dict["last_message_date"] == now
    assert data_dict["finished_downloading"] is False


def test_dict_initialization():
    # Test initialization from dictionary representation
    entity_dict = {
        "_": "Channel",
        "id": 12345,
        "title": "Test Channel Dict",
        "username": "testchanneldict",
        "participants_count": 1000,
        "megagroup": False,
        "creator": True
    }
    
    now = datetime.now(timezone.utc)
    
    # Mock the load_entity method
    original_load_entity = ChatData.load_entity
    
    def mock_load_entity(cls, entity_type, entity_data):
        entity = Mock()
        entity.id = entity_data["id"]
        entity.title = entity_data["title"]
        entity.username = entity_data["username"]
        entity.participants_count = entity_data["participants_count"]
        entity.megagroup = entity_data["megagroup"]
        entity.creator = entity_data["creator"]
        entity.__class__.__name__ = entity_type
        return entity
    
    try:
        ChatData.load_entity = classmethod(mock_load_entity)
        
        chat_data = ChatData(entity_dict, now)
        
        assert chat_data.id == 12345
        assert chat_data.entity.title == "Test Channel Dict"
        assert chat_data.entity.username == "testchanneldict"
        assert chat_data.last_message_date == now
    finally:
        # Restore original method
        ChatData.load_entity = original_load_entity