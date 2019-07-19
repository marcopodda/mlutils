import torch

from utils.module_loading import import_string
from utils.training import get_device

from .events import EventDispatcher
from .state import State
from .optimizer import Optimizer
from .monitor import Monitor
from .loggers import CSVLogger

from pathlib import Path


def build_model(config, dim_input, dim_target):
    model_config = config.get('model')
    model_class = import_string(model_config.class_name)
    model = model_class(dim_input, dim_target, **model_config.params)
    return model


def build_criterion(config):
    criterion_config = config.get('criterion')
    criterion_class = import_string(criterion_config.class_name)
    criterion = criterion_class(**criterion_config.params)
    return criterion


class Engine(EventDispatcher):
    def __init__(self, config, dim_input, dim_target, save_path):
        super().__init__()
        self.config = config
        self.default_device = get_device(config)

        self.state = State(
            model=build_model(config, dim_input, dim_target),
            criterion=build_criterion(config))

        # callbacks
        self.register(Optimizer(config, self.model))

        if 'monitor' in config:
            self.register(Monitor(config))

        if 'logger' in config:
            self.register(CSVLogger())

        self.save_path = save_path

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

        start_epoch = self.state.epoch + 1 if 'epoch' in self.state else 0
        num_epochs = num_epochs or self.config.max_epochs

        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.state.update(epoch=epoch)

            self.state.init_epoch_results()
            self._dispatch('on_epoch_start', self.state)

            self.set_training_mode()
            self.set_device(train_device)

            self._dispatch('on_training_epoch_start', self.state)
            self._train_one_epoch(train_loader)
            self._dispatch('on_training_epoch_end', self.state)

            if val_loader is not None:
                self.set_validation_mode()
                self.set_device(val_device)

                self._dispatch('on_validation_epoch_start', self.state)
                self._validate_one_epoch(val_loader)
                self._dispatch('on_validation_epoch_end', self.state)

            self._dispatch('on_epoch_end', self.state)
            self.state.save_epoch_results()

        self._dispatch('on_fit_end', self.state)

    def _train_one_epoch(self, loader):
        for idx, batch in enumerate(loader):
            self._dispatch('on_training_batch_start', self.state)

            train_data = self.feed_forward_batch(batch)
            self.state.update(**train_data)

            train_data['loss'].backward()
            self._dispatch('on_backward', self.state)

            self._dispatch('on_training_batch_end', self.state)

    def _validate_one_epoch(self, loader):
        for idx, batch in enumerate(loader):
            self._dispatch('on_validation_batch_start', self.state)
            self.state.update(**self.feed_forward_batch(batch))
            self._dispatch('on_validation_batch_end', self.state)

    def evaluate(self, test_loader, test_device=None):
        try:
            self.load(self.save_path, best=True)
        except FileNotFoundError:
            self.load(self.save_path, best=False)

        self.set_test_mode()
        self.set_device(test_device)

        self.state.init_epoch_results()
        self._dispatch('on_test_start', self.state)
        self._test_epoch(test_loader)
        self._dispatch('on_test_end', self.state)
        self.state.save_epoch_results()

    def _test_epoch(self, loader):
        for idx, batch in enumerate(loader):
            self._dispatch('on_test_batch_start', self.state)
            self.state.update(**self.feed_forward_batch(batch))
            self._dispatch('on_test_batch_end', self.state)

    def feed_forward_batch(self, batch):
        raise NotImplementedError

    def save(self, path, best=False):
        state_dict = self.state.state_dict()
        for event_handler in self._event_handlers:
            state_dict.update(event_handler.state_dict())

        filename = 'best.pt' if best else 'last.pt'
        torch.save(state_dict, path / filename)

    def load(self, path, best=False):
        filename = 'best.pt' if best else 'last.pt'
        state_dict = torch.load(path / filename)
        self.state.load_state_dict(state_dict)

        for event_handler in self._event_handlers:
            event_handler.load_state_dict(state_dict)