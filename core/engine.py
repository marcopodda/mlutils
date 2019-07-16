import torch

from utils.module_loading import import_string
from .events import EventDispatcher, State


class Engine(EventDispatcher):
    def __init__(self, config, dim_input, dim_target, event_handlers=[]):
        super().__init__(event_handlers=event_handlers)

        self.config = config
        self.model = import_string(config.model_class)(config, dim_input, dim_target)
        self.criterion = import_string(config.criterion_class)(config, dim_target)
        state_dict = {
            'config': config,
            'model': self.model,
            'criterion': self.criterion,
            'current_epoch': 0
        }

        self.state = State(**state_dict)

    def fit(self, train_loader, val_loader=None):
        self._dispatch('on_fit_start', self.state)

        for epoch in range(self.state.current_epoch, self.config.max_epochs):
            self._dispatch('on_epoch_start', self.state)

            self.model.train()
            self._dispatch('on_training_epoch_start', self.state)
            self._train(epoch, train_loader)
            self._dispatch('on_training_epoch_end', self.state)

            if val_loader is not None:
                self.model.eval()
                self._dispatch('on_validation_epoch_start', self.state)
                self._validate(epoch, val_loader)
                self._dispatch('on_validation_epoch_end', self.state)

                if self.state.stop_training:
                    break

            self._dispatch('on_epoch_end', self.state)

        self._dispatch('on_fit_end', self.state)

    def _train(self, epoch, loader):
        for idx, batch in enumerate(loader):
            self._dispatch('on_training_batch_start', self.state)

            train_dict = self.process_batch(batch)
            self.state.update(num_batches=len(loader), **train_dict)

            # train_dict['loss'].backward()
            self._dispatch('on_backward', self.state)

            self._dispatch('on_training_batch_end', self.state)

    def _validate(self, epoch, loader):
        for idx, batch in enumerate(loader):
            self._dispatch('on_validation_batch_start', self.state)

            val_dict = self.process_batch(batch)
            self.state.update(num_batches=len(loader), **val_dict)

            self._dispatch('on_validation_batch_end', self.state)

    def process_batch(self, batch):
        raise NotImplementedError