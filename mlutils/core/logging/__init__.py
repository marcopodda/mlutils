from loguru import logger

from pathlib import Path


def filter_record(*levels):
    def _filter(record):
        if record['level'] in levels:
            return record
    return _filter


class Logger(object):
    _instance = None

    def __new__(cls, log_dir=None):
        if Logger._instance is None:
            options = {
                "handlers": [
                    # {
                    #     "sink": open(os.devnull, 'w'),
                    #     "enqueue": True,
                    #     "format": '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | {message}'
                    # },
                    # {
                    #     "sink": sys.stdout,
                    #     "enqueue": True,
                    #     "format": '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | {message}'
                    # },
                    {
                        "sink": Path(log_dir) / 'performance.log',
                        "enqueue": True,
                        "format": '{message}',
                        "filter": filter_record("SUCCESS")
                    },
                    {
                        "sink": Path(log_dir) / "events.log",
                        "enqueue": True,
                        "filter": filter_record("INFO", "WARNING")
                    },
                ]
            }
            logger.configure(**options)
            Logger._instance = logger
        return Logger._instance
