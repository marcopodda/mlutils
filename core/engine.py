import torch

from core.optimizer import Optimizer
from core.monitors import Monitor
from core.saver import Saver
from utils.module_loading import import_string
from utils.training import get_device

from .events import EventDispatcher, State


def build_model(config, dim_input, dim_target):
    model_config = config.get('model')
    model_class = import_string(model_config.class_name)
    model = model_class(dim_input, dim_target, **model_config.params)
    device = get_device(config)
    return model.to(device)

def build_criterion(config):
    criterion_config = config.get('criterion')
    criterion_class = import_string(criterion_config.class_name)
    criterion = criterion_class(**criterion_config.params)
    device = get_device(config)
    return criterion.to(device)


class Engine(EventDispatcher):
    def __init__(self, config, dim_input, dim_target, load_path=None, save_path=None, best=False):
        super().__init__()
        self.config = config

        self.state = State(
            model=build_model(config, dim_input, dim_target),
            criterion=build_criterion(config),
            stop_training=False,
            epoch=0)

        # optimizer
        self.register(Optimizer(config.get('optimizer'), self.state.model))
        self.register(Monitor(config.get('monitor')))
        self.register(Saver(config.get('saver')))

        # gradient clipping
        if 'gradient_clipping' in config:
            grad_clip_config = config.get('gradient_clipping')
            grad_clip_class = import_string(grad_clip_config.class_name)
            grad_clipper = grad_clip_class(**grad_clip_config.params)
            self.register(grad_clipper)

        # early stopping
        if 'early_stopping' in config:
            early_stopper_config = config.get('early_stopping')
            early_stopper_class = import_string(early_stopper_config.class_name)
            early_stopper = early_stopper_class(**early_stopper_config.params)
            self.register(early_stopper)

        if load_path is not None:
            filename = 'best.pt' if best else 'last.pt'
            self.state.load(load_path / filename)

        self.save_path = save_path

    def fit(self, train_loader, val_loader=None, max_epochs=None):
        self._dispatch('on_fit_start', self.state)

        max_epochs = max_epochs or self.config.max_epochs
        for epoch in range(self.state.epoch, max_epochs):
            self.state.update(epoch=epoch)
            self._dispatch('on_epoch_start', self.state)

            self.state.model.train()
            self._dispatch('on_training_epoch_start', self.state)
            self._train_one_epoch(train_loader)
            self._dispatch('on_training_epoch_end', self.state)

            if val_loader is not None:
                self.state.model.eval()
                self._dispatch('on_validation_epoch_start', self.state)
                self._validate_one_epoch(val_loader)
                self._dispatch('on_validation_epoch_end', self.state)

            self._dispatch('on_epoch_end', self.state)

            if self.state.stop_training:
                print(f'Early stopping at epoch {epoch}.')
                break

            if self.save_path is not None:
                if self.state.save_best:
                    self.save(self.save_path, best=True)
                self.save(self.save_path, best=False)

        self._dispatch('on_fit_end', self.state)

    def _train_one_epoch(self, loader):
        for idx, batch in enumerate(loader):
            self._dispatch('on_training_batch_start', self.state)

            train_data = self.process_batch(batch)
            self.state.update(**train_data)

            train_data['loss'].backward()
            self._dispatch('on_backward', self.state)

            self._dispatch('on_training_batch_end', self.state)

    def _validate_one_epoch(self, loader):
        for idx, batch in enumerate(loader):
            self._dispatch('on_validation_batch_start', self.state)
            self.state.update(**self.process_batch(batch))
            self._dispatch('on_validation_batch_end', self.state)

    def evaluate(self, test_loader):
        self._dispatch('on_test_start', self.state)
        for idx, batch in enumerate(test_loader):
            self._dispatch('on_test_batch_start', self.state)
            self.state.update(**self.process_batch(batch))
            self._dispatch('on_test_batch_end', self.state)
        self._dispatch('on_test_end', self.state)

    def process_batch(self, batch):
        raise NotImplementedError

    def save(self, path, best=False):
        state_dict = self.state.state_dict()
        for event_handler in self._event_handlers:
            state_dict.update(event_handler.state_dict())

        filename = 'best.pt' if best else 'last.pt'
        torch.save(state_dict, path / filename)