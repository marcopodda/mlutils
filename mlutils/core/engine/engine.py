from mlutils.core.engine.callback.optimizer import Optimizer
from mlutils.core.engine.callback.monitor import Monitor
from mlutils.core.event.dispatcher import EventDispatcher
from mlutils.core.event.state import State
from mlutils.core.logging import logger
from mlutils.util.module_loading import load_class
from mlutils.util.training import get_device


class Engine(EventDispatcher):
    def __init__(self, config, dim_input, dim_target):
        super().__init__()
        self.config = config
        self.default_device = get_device(config)

        self.state = State(
            model=load_class(config.model, dim_input=dim_input, dim_target=dim_target),
            criterion=load_class(config.criterion))

        # callbacks
        self.register(Optimizer(config.optimizer, self.model))

        # metrics
        additional_metrics = []
        if 'callbacks' in config:
            if 'metrics' in config.callbacks:
                for metric_config in config.callbacks.metrics:
                    metric = load_class(metric_config)
                    additional_metrics.append(metric)
        self.register(Monitor(additional_metrics))

        if 'callbacks' in config:
            if 'early_stopper' in config.callbacks:
                early_stopper = load_class(config.callbacks.early_stopper)
                self.register(early_stopper)

            if 'model_saver' in config.callbacks:
                model_saver = load_class(config.callbacks.modelsaver)
                self.register(model_saver)

            if 'loggers' in config.callbacks:
                for logger_config in config.callbacks.loggers:
                    logger = load_class(logger_config)
                    self.register(logger)

    @property
    def model(self):
        return self.state.model

    @property
    def criterion(self):
        return self.state.criterion

    def set_device(self, device):
        self.device = device or self.default_device
        self.model.to(self.device)
        self.criterion.to(self.device)

    def set_training_mode(self):
        self.model.train()
        self.criterion.train()
        self.state.update(phase="training")

    def set_validation_mode(self):
        self.model.eval()
        self.criterion.eval()
        self.state.update(phase="validation")

    def set_test_mode(self):
        self.model.eval()
        self.criterion.eval()
        self.state.update(phase="test")

    def fit(self, train_loader, val_loader=None, num_epochs=None, train_device=None, val_device=None):
        self._dispatch('on_fit_start', self.state)

        start_epoch = self.state.epoch
        num_epochs = num_epochs or self.config.max_epochs
        logger.info(f'{"Starting" if start_epoch == 0 else "Resuming"} training at epoch: {self.state.epoch}')

        for epoch in range(start_epoch, start_epoch + num_epochs):
            if self.state.stop_training is True:
                logger.info(f"Early stopping at epoch {self.state.epoch}.")
                break

            self.state.update(epoch=epoch)

            logger.info(f"Starting epoch {self.state.epoch}.")
            self._dispatch('on_epoch_start', self.state)

            self.set_training_mode()
            logger.info("Training mode set.")

            self.set_device(train_device)
            logger.info(f"Device is now '{self.device}'.")

            self._dispatch('on_training_epoch_start', self.state)
            self._train_epoch(train_loader)
            self._dispatch('on_training_epoch_end', self.state)

            if val_loader is not None:
                self.set_validation_mode()
                logger.info("Validation mode set.")

                self.set_device(val_device)
                logger.info(f"Device is now '{self.device}'.")

                self._dispatch('on_validation_epoch_start', self.state)
                self._evaluate_epoch(val_loader)
                self._dispatch('on_validation_epoch_end', self.state)

            self._dispatch('on_epoch_end', self.state)
            self.state.save_epoch_results()

        self._dispatch('on_fit_end', self.state)

    def evaluate(self, test_loader, path, test_device=None):
        logger.info("Starting final evaluation.")

        try:
            self.load(path, best=True)
        except FileNotFoundError:
            logger.warning("Best model not found. Loading last model.")
            self.load(path, best=False)

        self.set_test_mode()
        self.set_device(test_device)

        self._dispatch('on_test_epoch_start', self.state)
        self._evaluate_epoch(test_loader)
        self._dispatch('on_test_epoch_end', self.state)
        self.state.save_epoch_results()

    def _train_epoch(self, loader):
        logger.info(f"Training epoch {self.state.epoch}.")
        for idx, batch in enumerate(loader):
            self._dispatch('on_training_batch_start', self.state)

            train_data = self.feed_forward_batch(batch)
            self.state.update(**train_data)

            train_data['loss'].backward()
            self._dispatch('on_backward', self.state)

            self._dispatch('on_training_batch_end', self.state)
        logger.info(f"Training finished.")

    def _evaluate_epoch(self, loader):
        logger.info(f"{self.state.phase.capitalize()} epoch {self.state.epoch}.")
        for idx, batch in enumerate(loader):
            self._dispatch(f'on_{self.state.phase}_batch_start', self.state)
            self.state.update(**self.feed_forward_batch(batch))
            self._dispatch(f'on_{self.state.phase}_batch_end', self.state)
        logger.info(f"{self.state.phase.capitalize()} finished.")

    def feed_forward_batch(self, batch):
        raise NotImplementedError

    def load(self, path, best=False):
        filename = 'best.pt' if best else 'last.pt'
        self.state.load(path / filename)
