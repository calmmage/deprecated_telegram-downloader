import sys
from datetime import datetime, timezone


def setup_logger(logger, level: str = "INFO"):
    logger.remove()  # Remove default handler
    logger.add(
        sink=sys.stderr,
        format="<level>{time:HH:mm:ss}</level> | <level>{message}</level>",
        colorize=True,
        level=level,
    )


def ensure_utc_datetime(dt: datetime | None) -> datetime | None:
    """Ensure datetime is timezone-aware in UTC.

    Args:
        dt: Input datetime, can be naive or timezone-aware

    Returns:
        datetime: Timezone-aware datetime in UTC, or None if input was None
    """
    if dt is None:
        return None

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
