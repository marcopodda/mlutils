import torch

from utils.module_loading import import_string
from utils.training import get_device

from .events import EventDispatcher, State
from .optimizer import Optimizer
from .monitor import Monitor
from .timer import Timer


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
    def __init__(self, config, dim_input, dim_target, save_path):
        super().__init__()
        self.config = config

        self.state = State(
            model=build_model(config, dim_input, dim_target),
            criterion=build_criterion(config),
            stop_training=False,
            save_best=False)

        # callbacks
        self.register(Optimizer(config, self.state.model))
        if 'monitor' in config:
            self.register(Monitor(config))
        if 'timer' in config:
            self.register(Timer())

        self.save_path = save_path

    def fit(self, train_loader, val_loader=None, num_epochs=None):
        self._dispatch('on_fit_start', self.state)

        start_epoch = self.state.epoch + 1 if 'epoch' in self.state else 0
        num_epochs = num_epochs or self.config.max_epochs

        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.state.update(epoch=epoch)
            print(epoch)
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
                    print(f'Saving best model at epoch {epoch}.')
                    self.save(self.save_path, best=True)
                self.save(self.save_path, best=False)

        self._dispatch('on_fit_end', self.state)
        self.save(self.save_path, best=False)

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
        try:
            self.load(self.save_path, best=True)
        except FileNotFoundError:
            self.load(self.save_path, best=False)

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
        print(state_dict['timer'])

    def load(self, path, best=False):
        filename = 'best.pt' if best else 'last.pt'
        state_dict = torch.load(path / filename)
        self.state.load_state_dict(state_dict)

        for event_handler in self._event_handlers:
            event_handler.load_state_dict(state_dict)
