from loguru import logger


def make_filter(name):
    def filter(record):
        return record["extra"].get("name") == name
    return filter


class Logger:
    def __init__(self, path, config):
        self.path = path

        self.event_logger = None
        self.performance_logger = None

        logger_config = config.get('logger')

        if logger_config.log_training:
            logger.add(self.path / "training.log", filter=make_filter("training"), format="{message}")
            self.training_logger = logger.bind(name="training")

        if logger_config.log_evaluation:
            logger.add(self.path / "evaluation.log", filter=make_filter("evaluation"), format="{message}")
            self.evaluation_logger = logger.bind(name="evaluation")

        if logger_config.log_events:
            logger.add(self.path / "events.log",
                        filter=make_filter("events"),
                        format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}")

    def log_event(self, msg, level="INFO"):
        if self.event_logger is not None:
            self.event_logger.log(level, msg)

    def log_training(self, msg):
        if self.training_logger is not None:
            self.training_logger.info(msg)

    def log_evaluation(self, msg):
        if self.evaluation_logger is not None:
            self.evaluation_logger.info(msg)