import asyncio
import os
from pathlib import Path
from typing import Dict

from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings
from telethon import TelegramClient

from telegram_downloader.config import StorageMode
from telegram_downloader.utils import setup_logger


class TelethonClientManagerEnvSettings(BaseSettings):
    TELEGRAM_API_ID: int = Field()
    TELEGRAM_API_HASH: str = Field()
    SESSIONS_DIR: Path = Field(default=Path("sessions"))
    MONGO_CONN_STR: str | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


class TelethonClientManager:
    def __init__(self, storage_mode: StorageMode = StorageMode.LOCAL, **kwargs):
        self.env = TelethonClientManagerEnvSettings(**kwargs)
        self.storage_mode = storage_mode
        self.api_id = self.env.TELEGRAM_API_ID
        self.api_hash = self.env.TELEGRAM_API_HASH
        self.sessions_dir = self.env.SESSIONS_DIR
        self.sessions_dir.mkdir(exist_ok=True)
        self.clients: Dict[int, TelegramClient] = {}
        logger.debug(f"TelethonManager initialized with storage_mode: {self.storage_mode}")
        logger.debug(f"TelethonManager initialized with sessions dir: {self.sessions_dir}")

    async def get_telethon_client(self, user_id: int) -> TelegramClient:
        logger.debug(
            f"Getting telethon client for user {user_id} with storage mode {self.storage_mode}"
        )
        if self.storage_mode == StorageMode.LOCAL:
            return await self._get_telethon_client_from_disk(user_id)
        elif self.storage_mode == StorageMode.MONGO:
            return await self._get_telethon_client_from_database(user_id)
        else:
            raise ValueError(f"Invalid storage mode: {self.storage_mode}")

    # region trajectory 1 save and load conn from disk
    async def _get_telethon_client_from_disk(self, user_id: int) -> TelegramClient:
        logger.debug(f"Attempting to get telethon client from disk for user {user_id}")
        if self._check_if_conn_is_present_on_disk(user_id):
            logger.debug(f"Found existing connection on disk for user {user_id}")
            return await self._load_conn_from_disk(user_id)
        else:
            logger.debug(f"No existing connection found on disk for user {user_id}, creating new")
            return await self._create_new_telethon_client_and_save_to_disk(user_id)

    # 2 - create new conn from scratch -> save conn to disk
    async def _create_new_telethon_client_and_save_to_disk(self, user_id: int) -> TelegramClient:
        session_key = self.sessions_dir / f"user_{user_id}"
        session_file = session_key.with_suffix(".session")
        logger.debug(f"Creating new telethon client for user {user_id} at {session_file}")

        client = TelegramClient(str(session_key), self.api_id, self.api_hash)
        try:
            logger.debug(f"Attempting to connect client for user {user_id}")
            await client.connect()

            # Check if already authorized
            if await client.is_user_authorized():
                logger.debug(f"Client already authorized for user {user_id}")
                self.clients[user_id] = client
                return client

            # Get phone number from environment
            phone = os.getenv("TELEGRAM_PHONE_NUMBER")
            if not phone:
                raise ValueError("TELEGRAM_PHONE_NUMBER environment variable is required")

            logger.debug(f"Sending code request for phone {phone}")
            send_code_result = await client.send_code_request(phone)

            logger.info("Please check your Telegram app and enter the verification code:")
            code = input("Code: ").strip()
            if not code:
                raise ValueError("Verification code is required")

            try:
                await client.sign_in(phone, code, phone_code_hash=send_code_result.phone_code_hash)
            except Exception as e:
                if "password" in str(e).lower():
                    # 2FA is enabled
                    password = os.getenv("TELEGRAM_2FA_PASSWORD")
                    if not password:
                        raise ValueError("2FA is enabled but TELEGRAM_2FA_PASSWORD not provided")
                    await client.sign_in(password=password)
                else:
                    raise

            if await client.is_user_authorized():
                logger.debug(f"Successfully authorized client for user {user_id}")
                self.clients[user_id] = client
                return client

            raise Exception("Failed to authorize client")

        except Exception as e:
            logger.error(f"Failed to create new client for user {user_id}: {e}")
            if session_file.exists():
                logger.debug(f"Removing failed session file for user {user_id}")
                session_file.unlink()
            raise

    # 3 - check if conn is present on disk
    def _check_if_conn_is_present_on_disk(self, user_id: int) -> bool:
        session_file = self.sessions_dir / f"user_{user_id}.session"
        exists = session_file.exists()
        logger.debug(f"Checking if session file exists for user {user_id}: {exists}")
        return exists

    # 4 - load conn from disk
    async def _load_conn_from_disk(self, user_id: int) -> TelegramClient:
        session_key = self.sessions_dir / f"user_{user_id}"
        session_file = session_key.with_suffix(".session")
        logger.debug(f"Loading session for user {user_id} from {session_file}")

        if not session_file.exists():
            logger.debug(f"No session file found for user {user_id}")
            raise Exception("No session file found")

        client = TelegramClient(str(session_key), self.api_id, self.api_hash)
        try:
            logger.debug(f"Attempting to connect existing client for user {user_id}")
            await client.connect()

            is_authorized = await client.is_user_authorized()
            logger.debug(f"Session authorization check for user {user_id}: {is_authorized}")

            if is_authorized:
                self.clients[user_id] = client
                logger.debug(f"Successfully loaded and authorized client for user {user_id}")
                return client

            logger.debug(f"Session exists but not authorized for user {user_id}")
            raise Exception("Session exists but not authorized")

        except Exception as e:
            logger.warning(f"Failed to load session for user {user_id}: {e}")
            raise e

    # endregion trajectory 1

    # region trajectory 2 save conn to db
    async def _get_telethon_client_from_database(self, user_id: int) -> TelegramClient:
        logger.debug(f"Attempting to get telethon client from database for user {user_id}")
        if await self._check_if_conn_is_present_in_db(user_id):
            logger.debug(f"Found existing connection in database for user {user_id}")
            return await self._load_conn_from_db(user_id)
        else:
            logger.debug(
                f"No existing connection found in database for user {user_id}, creating new"
            )
            return await self._create_new_telethon_client_and_save_to_db(user_id)

    # 5 - create new conn from scratch -> save conn to db
    async def _create_new_telethon_client_and_save_to_db(self, user_id: int) -> TelegramClient:
        # This is a placeholder - you'll need to implement the database storage logic
        logger.debug("Database storage not implemented yet")
        raise NotImplementedError("Database storage not implemented yet")

    # 6 - load conn from db
    async def _load_conn_from_db(self, user_id: int) -> TelegramClient:
        # This is a placeholder - you'll need to implement the database loading logic
        logger.debug("Database loading not implemented yet")
        raise NotImplementedError("Database loading not implemented yet")

    # 7 - check if conn is present in db
    async def _check_if_conn_is_present_in_db(self, user_id: int) -> bool:
        # This is a placeholder - you'll need to implement the database check logic
        logger.debug("Database check not implemented yet")
        raise NotImplementedError("Database check not implemented yet")

    # endregion trajectory 2

    # async def disconnect_all(self):
    #     """Disconnect all clients"""
    #     logger.debug(f"Disconnecting all clients ({len(self.clients)} total)")
    #     for client in self.clients.values():
    #         await client.disconnect()
    #     self.clients.clear()
    #     logger.debug("All clients disconnected")


async def main(debug: bool = False):
    from dotenv import load_dotenv

    load_dotenv()
    setup_logger(logger, level="DEBUG" if debug else "INFO")

    # Get required environment variables
    api_id = os.getenv("TELEGRAM_API_ID")
    if not api_id:
        raise ValueError("TELEGRAM_API_ID environment variable is required")

    api_hash = os.getenv("TELEGRAM_API_HASH")
    if not api_hash:
        raise ValueError("TELEGRAM_API_HASH environment variable is required")

    user_id = os.getenv("TELEGRAM_USER_ID")
    if not user_id:
        raise ValueError("TELEGRAM_USER_ID environment variable is required")

    sessions_dir = Path("sessions")
    sessions_dir.mkdir(exist_ok=True)

    telethon_manager = TelethonClientManager(
        storage_mode=StorageMode.LOCAL,
        TELEGRAM_API_ID=int(api_id),
        TELEGRAM_API_HASH=api_hash,
        SESSIONS_DIR=sessions_dir,
    )

    # Get client for user
    try:
        client = await telethon_manager.get_telethon_client(int(user_id))
    except Exception as e:
        logger.error(f"Failed to get client: {e}")
        return

    # Example: Get first dialog and message
    try:
        async for dialog in client.iter_dialogs(limit=1):
            print("\nChat details:")
            print(f"Name: {dialog.name}")
            print(f"ID: {dialog.id}")
            print(f"Type: {dialog.entity.__class__.__name__}")

            # Get first message from this dialog
            async for message in client.iter_messages(dialog.id, limit=1):
                print("\nLatest message:")
                print(f"From: {message.sender_id}")
                print(f"Date: {message.date}")
                print(f"Text: {message.text}")
    except Exception as e:
        logger.error(f"Error accessing messages: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Run the async main function
    asyncio.run(main(debug=args.debug))
