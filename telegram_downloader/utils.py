import sys


def setup_logger(logger, level: str = "INFO"):
    logger.remove()  # Remove default handler
    logger.add(
        sink=sys.stderr,
        format="<level>{time:HH:mm:ss}</level> | <level>{message}</level>",
        colorize=True,
        level=level,
    )
