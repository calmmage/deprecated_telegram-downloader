import pytest
from typing import get_type_hints

from telegram_downloader.telegram_downloader import TelegramDownloader


def test_init_type_hints():
    """Test that the __init__ method of TelegramDownloader has correct type hints."""
    hints = get_type_hints(TelegramDownloader.__init__)
    
    # Check required parameters
    assert "config_path" in hints
    assert "db" in hints
    assert "telethon_client" in hints
    assert "return" in hints  # Return type hint should be present
    
    # Check parameter types (string representation since actual types may vary)
    str_hints = {k: str(v) for k, v in hints.items()}
    
    # config_path should accept Path or str
    assert "Path | str" in str_hints["config_path"] or "str | Path" in str_hints["config_path"]
    
    # db should be optional Database
    assert "Optional" in str_hints["db"] or "None" in str_hints["db"]
    
    # telethon_client should be optional TelegramClient
    assert "Optional" in str_hints["telethon_client"] or "None" in str_hints["telethon_client"]
    
    # Return type should be None
    assert str_hints["return"] == "None" or str_hints["return"] == "<class 'NoneType'>"


def test_important_method_type_hints():
    """Test that important methods of TelegramDownloader have correct type hints."""
    # Check get_telethon_client
    hints = get_type_hints(TelegramDownloader.get_telethon_client)
    assert "return" in hints
    assert "TelegramClient" in str(hints["return"])
    
    # Check _get_chats
    hints = get_type_hints(TelegramDownloader._get_chats)
    assert "return" in hints
    assert "List" in str(hints["return"]) and "ChatData" in str(hints["return"])
    
    # Check main method
    hints = get_type_hints(TelegramDownloader.main)
    assert "ignore_finished" in hints
    assert "chat_sample_size" in hints
    assert "bool" in str(hints["ignore_finished"])
    assert "None" in str(hints["chat_sample_size"]) or "Optional" in str(hints["chat_sample_size"])