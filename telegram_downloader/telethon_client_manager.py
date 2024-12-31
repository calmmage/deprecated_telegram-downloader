import asyncio
import os
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from loguru import logger
from telethon import TelegramClient

from telegram_downloader.utils import setup_logger


class StorageMode(str, Enum):
    TO_DATABASE = "to_database"
    TO_DISK = "to_disk"


class TelethonClientManager:
    def __init__(self, storage_mode: StorageMode, api_id: int, api_hash: str, sessions_dir: Path):
        self.storage_mode = storage_mode
        self.api_id = api_id
        self.api_hash = api_hash
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(exist_ok=True)
        self.clients: Dict[int, TelegramClient] = {}
        logger.debug(f"TelethonManager initialized with storage_mode: {storage_mode}")
        logger.debug(f"TelethonManager initialized with sessions dir: {sessions_dir}")

    async def get_telethon_client(self, user_id: int) -> TelegramClient:
        logger.debug(
            f"Getting telethon client for user {user_id} with storage mode {self.storage_mode}"
        )
        if self.storage_mode == StorageMode.TO_DISK:
            return await self._get_telethon_client_from_disk(user_id)
        elif self.storage_mode == StorageMode.TO_DATABASE:
            return await self._get_telethon_client_from_database(user_id)

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
    async def _create_new_telethon_client_and_save_to_disk(
        self, user_id: int
    ) -> Optional[TelegramClient]:
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

            # Get phone number from environment or ask user
            phone = os.getenv("TELEGRAM_PHONE_NUMBER")
            if not phone:
                logger.error("No phone number provided in environment")
                return None

            logger.debug(f"Sending code request for phone {phone}")
            # Send code request
            send_code_result = await client.send_code_request(phone)

            # Get verification code from environment or ask user
            code = os.getenv("TELEGRAM_CODE")
            if not code:
                logger.error("No verification code provided in environment")
                return None

            logger.debug("Attempting to sign in with code")
            try:
                # Try to sign in with the code
                await client.sign_in(phone, code, phone_code_hash=send_code_result.phone_code_hash)
            except Exception as e:
                if "password" in str(e).lower():
                    # 2FA is enabled, get password from environment
                    password = os.getenv("TELEGRAM_2FA_PASSWORD")
                    if not password:
                        logger.error("2FA is enabled but no password provided in environment")
                        return None

                    logger.debug("2FA enabled, attempting to sign in with password")
                    # Sign in with password
                    await client.sign_in(password=password)
                else:
                    raise

            # Check if we're now authorized
            if await client.is_user_authorized():
                logger.debug(f"Successfully authorized client for user {user_id}")
                self.clients[user_id] = client
                return client

            logger.debug(f"Failed to authorize client for user {user_id}")
            raise Exception("Failed to authorize client")

        except Exception as e:
            logger.warning(f"Failed to create new client for user {user_id}: {e}")
            if session_file.exists():
                logger.debug(f"Removing failed session file for user {user_id}")
                session_file.unlink()
            raise e

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
    async def _get_telethon_client_from_database(self, user_id: int) -> Optional[TelegramClient]:
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
    async def _create_new_telethon_client_and_save_to_db(
        self, user_id: int
    ) -> Optional[TelegramClient]:
        # This is a placeholder - you'll need to implement the database storage logic
        logger.debug("Database storage not implemented yet")
        raise NotImplementedError("Database storage not implemented yet")

    # 6 - load conn from db
    async def _load_conn_from_db(self, user_id: int) -> Optional[TelegramClient]:
        # This is a placeholder - you'll need to implement the database loading logic
        logger.debug("Database loading not implemented yet")
        raise NotImplementedError("Database loading not implemented yet")

    # 7 - check if conn is present in db
    async def _check_if_conn_is_present_in_db(self, user_id: int) -> bool:
        # This is a placeholder - you'll need to implement the database check logic
        logger.debug("Database check not implemented yet")
        raise NotImplementedError("Database check not implemented yet")

    # endregion trajectory 2

    async def disconnect_all(self):
        """Disconnect all clients"""
        logger.debug(f"Disconnecting all clients ({len(self.clients)} total)")
        for client in self.clients.values():
            await client.disconnect()
        self.clients.clear()
        logger.debug("All clients disconnected")


async def main(debug: bool = False):
    from dotenv import load_dotenv

    load_dotenv()
    setup_logger(logger, level="DEBUG" if debug else "INFO")

    # Example initialization:
    SESSIONS_DIR = Path("sessions")
    SESSIONS_DIR.mkdir(exist_ok=True)
    TELEGRAM_API_ID = int(os.getenv("TELEGRAM_API_ID"))
    TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH")
    telethon_manager = TelethonClientManager(
        storage_mode=StorageMode.TO_DISK,
        api_id=TELEGRAM_API_ID,
        api_hash=TELEGRAM_API_HASH,
        sessions_dir=SESSIONS_DIR,
    )

    user_id = os.getenv("TELEGRAM_USER_ID")

    # Get client for user
    client = await telethon_manager.get_telethon_client(int(user_id))
    if not client:
        print("Failed to get client")
        return

    # Get one random chat/contact
    async for dialog in client.iter_dialogs():
        print("\nRandom chat details:")
        print(f"Chat name: {dialog.name}")
        print(f"Chat ID: {dialog.id}")
        print(f"Chat type: {dialog.entity.__class__.__name__}")
        break  # Just get the first one

    # Get one random message
    async for message in client.iter_messages(dialog):
        print("\nRandom message:")
        print(f"From: {message.sender_id}")
        print(f"Date: {message.date}")
        print(f"Text: {message.text}")
        break  # Just get the first one


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Run the async main function
    asyncio.run(main(debug=args.debug))
