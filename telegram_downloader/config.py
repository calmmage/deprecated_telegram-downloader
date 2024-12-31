from pathlib import Path

import yaml
from loguru import logger
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class EnvSettings(BaseSettings):
    # Telegram API credentials
    TELEGRAM_API_ID: str | None = None
    TELEGRAM_API_HASH: str | None = None

    # Optional database connection if needed
    MONGO_CONN_STR: str | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# class DownloadConfig(BaseModel):
#     max_messages: int
#     file_types: list[str]
#     save_path: str

# class FilterConfig(BaseModel):
#     date_from: str | None
#     date_to: str | None
#     min_file_size: int
#     max_file_size: int | None

# class LoggingConfig(BaseModel):
#     level: str
#     file: str


class AppConfig(BaseModel):
    # download: DownloadConfig
    # filters: FilterConfig
    # logging: LoggingConfig

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        with path.open("r") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
