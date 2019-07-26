from loguru import logger

from pathlib import Path


def filter_record(*levels):
    def _filter(record):
        if record['level'] in levels:
            return record
    return _filter

def get_logger_options(log_dir):
    return {
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

class Logger(object):
    _instance = None

    def __new__(cls, log_dir=None):
        if log_dir is None:
            return Logger._instance

        options = get_logger_options(log_dir)

        if Logger._instance is None:
            logger.configure(**options)
            Logger._instance = logger
        else:
            Logger._instance.configure(**options)

        return Logger._instance
