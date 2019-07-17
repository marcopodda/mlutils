import torch

from core.optimizer import Optimizer
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
    def __init__(self, config, dim_input, dim_target, path=None):
        super().__init__()
        self.config = config

        self.model = build_model(config, dim_input, dim_target)
        self.criterion = build_criterion(config)

        if path is not None:
            self.model.load_state_dict(torch.load(path / 'model.pt'))
            self.criterion.load_state_dict(torch.load(path / 'criterion.pt'))

        self.register(Optimizer(config, self.model, path=path))
        for entry in config.get('event_handlers'):
            eh_class = import_string(entry.class_name)
            eh = eh_class(path=path, **entry.params)
            self.register(eh)

        for entry in config.get('metrics'):
            eh_class = import_string(entry.class_name)
            eh = eh_class(path=path, **entry.params)
            self.register(eh)

        state_dict = {
            'config': config,
            'model': self.model,
            'criterion': self.criterion,
        }

        self.state = State(**state_dict)

    def fit(self, train_loader, val_loader=None, max_epochs=None):
        self._dispatch('on_fit_start', self.state)

        max_epochs = max_epochs or self.config.max_epochs
        for epoch in range(max_epochs):
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

            train_data = self.process_batch(batch)
            self.state.update(**train_data)

            train_data['loss'].backward()
            self._dispatch('on_backward', self.state)

            self._dispatch('on_training_batch_end', self.state)

    def _validate(self, epoch, loader):
        for idx, batch in enumerate(loader):
            self._dispatch('on_validation_batch_start', self.state)

            self.state.update(**self.process_batch(batch))

            self._dispatch('on_validation_batch_end', self.state)

    def process_batch(self, batch):
        raise NotImplementedError

    def save(self, path):
        torch.save(self.model.state_dict(), path / 'model.pt')
        torch.save(self.criterion.state_dict(), path / 'criterion.pt')
        self._dispatch('save', path)