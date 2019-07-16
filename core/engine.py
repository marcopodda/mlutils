import torch
from torch import optim
from .events import EventDispatcher, State


def load_model(config, dim_input, dim_target, ckpt):
    model = config.model_class(dim_input, dim_target, **config.model_params)
    if ckpt is not None:
        model.load_state_dict(ckpt['model'])
    return model


def load_criterion(config, dim_target, ckpt):
    criterion = config.criterion_class(dim_target, **config.criterion_params)
    if ckpt is not None:
        criterion.load_state_dict(ckpt['criterion'])
    return criterion

def load_optimizer(config, model):
    optimizer_class = getattr(optim, config.optimizer_name)
    return optimizer_class(model.parameters(), **config.optimizer_params)


def load_state(ckpt):
    state = State()
    if ckpt:
        state.update(**ckpt['state'])
    return state


class Engine(EventDispatcher):
    def __init__(self, config, model_class, criterion_class, dim_input, dim_target, event_handlers=[]):
        super().__init__(event_handlers=event_handlers)

        self.config = config
        self.model = model_class(config, dim_input, dim_target)
        self.criterion = criterion_class(config, dim_target)
        self.optimizer = load_optimizer(config, self.model)
        self.state = State()

    def fit(self, train_loader, val_loader=None):
        self.state.update(model=self.model, optimizer=self.optimizer)
        self._dispatch('on_fit_start', self.state)

        for epoch in range(self.state.epoch, self.config.max_epochs):
            self.state.update(epoch=epoch)

            self.model.train()
            self.state.update(training=self.model.training)

            self._dispatch('on_epoch_start', self.state)
            self._run_epoch(epoch, train_loader)
            self.state.update(num_batches=len(train_loader))
            self._dispatch('on_epoch_end', self.state)

            if val_loader is not None:
                self.model.eval()
                self.state.update(training=self.model.training)

                self._dispatch('on_epoch_start', self.state)
                self._run_epoch(epoch, val_loader)
                self.state.update(num_batches=len(val_loader))
                self._dispatch('on_epoch_end', self.state)

                if self.state.stop_training:
                    break

        self._dispatch('on_fit_end', self.state)

    def _run_epoch(self, epoch, loader):
        for idx, batch in enumerate(loader):
            self._dispatch('on_batch_start', self.state)

            if self.model.training:
                self.optimizer.zero_grad()

            loss, outputs, targets = self.process_batch(batch)
            epoch_loss = loss.item() / len(loader)
            self.state.update(loss=epoch_loss, outputs=outputs, targets=targets)

            if self.model.training:
                loss.backward()
                self._dispatch('on_parameter_update', self.state)
                self.optimizer.step()

            self._dispatch('on_batch_end', self.state)

    def process_batch(self, batch):
        raise NotImplementedError