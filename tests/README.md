# Tests for Telegram Downloader

This directory contains the test suite for the Telegram Downloader project. The tests use `pytest`, `pytest-asyncio`, and `pytest-mock` to provide comprehensive test coverage for the codebase.

## Running Tests

To run the tests, make sure you have all the required dependencies installed:

```bash
poetry install
```

Then, run the tests using pytest:

```bash
# Run all tests
poetry run pytest

# Run tests with coverage report
poetry run pytest --cov=telegram_downloader

# Run a specific test file
poetry run pytest tests/test_data_model.py

# Run tests matching a specific pattern
poetry run pytest -k "config"
```

## Test Structure

The test suite is structured as follows:

- `test_data_model.py`: Tests for the ChatData and other data model classes
- `test_config.py`: Tests for configuration loading and validation
- `test_utils.py`: Tests for utility functions
- `test_telethon_client_manager.py`: Tests for the Telethon client manager
- `test_telegram_downloader.py`: Tests for the main TelegramDownloader class
- `test_integration.py`: Integration tests for the entire workflow
- `test_imports.py`: Basic import tests to ensure the package is importable

## Writing New Tests

When writing new tests, please follow these guidelines:

1. Use appropriate fixtures to set up test data
2. Use mocks for external dependencies like Telethon and MongoDB
3. Use descriptive test names that explain what functionality is being tested
4. Mark slow or integration tests appropriately using pytest.mark
5. Use async tests with pytest.mark.asyncio for testing async functions

## Coverage

To generate a coverage report, run:

```bash
poetry run pytest --cov=telegram_downloader --cov-report=html
```

This will generate an HTML coverage report in the `htmlcov` directory.