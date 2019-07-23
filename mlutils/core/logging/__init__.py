import sys
from loguru import logger

from pathlib import Path


config = {
    "handlers": [
        # {"sink": open(os.devnull, 'w'), "enqueue": True, "format": '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | {message}'},
        {"sink": sys.stdout, "enqueue": True, "format": '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | {message}'},
        {"sink": Path('mlutils/logs') / "events" / "events_{time}.log", "enqueue": True},
    ]
}

logger.configure(**config)