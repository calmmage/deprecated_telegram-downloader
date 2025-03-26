import sys
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from loguru import logger

from telegram_downloader.utils import ensure_utc_datetime, setup_logger


def test_ensure_utc_datetime_with_none():
    # Test with None input
    result = ensure_utc_datetime(None)
    assert result is None


def test_ensure_utc_datetime_with_naive_datetime():
    # Test with naive datetime (no timezone info)
    naive_dt = datetime(2023, 1, 1, 12, 0, 0)
    result = ensure_utc_datetime(naive_dt)
    
    # Result should be timezone-aware with UTC timezone
    assert result.tzinfo is not None
    assert result.tzinfo == timezone.utc
    assert result == datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_ensure_utc_datetime_with_other_timezone():
    # Test with datetime in a different timezone
    est = timezone(offset=datetime.timedelta(hours=-5))
    est_dt = datetime(2023, 1, 1, 7, 0, 0, tzinfo=est)  # 7 AM EST = 12 PM UTC
    
    result = ensure_utc_datetime(est_dt)
    
    # Result should be converted to UTC
    assert result.tzinfo == timezone.utc
    assert result == datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_ensure_utc_datetime_with_already_utc():
    # Test with datetime already in UTC
    utc_dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    result = ensure_utc_datetime(utc_dt)
    
    # Result should be the same
    assert result == utc_dt
    assert result.tzinfo == timezone.utc


@patch("loguru.logger.remove")
@patch("loguru.logger.add")
def test_setup_logger_default_level(mock_add, mock_remove):
    # Test with default INFO level
    setup_logger(logger)
    
    # Check that logger.remove() was called
    mock_remove.assert_called_once()
    
    # Check that logger.add() was called with proper parameters
    mock_add.assert_called_once()
    args, kwargs = mock_add.call_args
    
    assert kwargs["sink"] == sys.stderr
    assert kwargs["level"] == "INFO"
    assert "format" in kwargs
    assert kwargs["colorize"] is True


@patch("loguru.logger.remove")
@patch("loguru.logger.add")
def test_setup_logger_debug_level(mock_add, mock_remove):
    # Test with DEBUG level
    setup_logger(logger, level="DEBUG")
    
    # Check that logger.remove() was called
    mock_remove.assert_called_once()
    
    # Check that logger.add() was called with proper parameters
    mock_add.assert_called_once()
    args, kwargs = mock_add.call_args
    
    assert kwargs["sink"] == sys.stderr
    assert kwargs["level"] == "DEBUG"
    assert "format" in kwargs
    assert kwargs["colorize"] is True