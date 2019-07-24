from torch.utils.data.dataloader import default_collate as torch_collate

from mlutils.settings import Settings


class Batch:
    def __init__(self, data):
        assert data != []
        self.settings = Settings()
        self._validate_keys(data)
        self.length = len(data)

        for key in self.keys:
            setattr(self, key, [])

        for pattern in data:
            for key in pattern:
                getattr(self, key).append(pattern[key])

        for key in self.keys:
            self[key] = torch_collate([d[key] for d in data])

    def _validate_keys(self, data):
        self.keys = list(data[0].keys())
        for pattern in data[1:]:
            if set(pattern.keys()) != set(self.keys):
                raise ValueError(f'Patterns have different keys: had {self.keys} before, but found {pattern.keys()}.')

    def __len__(self):
        return self.length

    def __getitem__(self, name):
        return getattr(self, name)

    def __setitem__(self, name, value):
        return setattr(self, name, value)

    def __iter__(self):
        return iter(self.__dict__.keys())

    @property
    def features(self):
        return self[self.settings.FEATURES_NAME]

    @property
    def target(self):
        return self[self.settings.TARGET_NAME]


def default_collate(batch_data):
    return Batch(batch_data)
