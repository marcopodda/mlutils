from pathlib import Path

from ..config import Config
from ..utils.os import get_or_create_dir
from ..data.splitters import CVHoldoutSplitter, HoldoutSplitter


ROOT = Path('RUNS')


class Experiment:
    @classmethod
    def load(cls, name, config_file):
        return cls(name, config_file)

    @property
    def name(self):
        if not hasattr(self, '_name'):
            self._name = "EXPERIMENT"
        return self._name

    def __init__(self, config_file):
        self.config = Config.from_file(config_file)
        self.root_folder = self.get_exp_root_folder()
        self.data_root_folder = self.get_exp_data_root_folder()

    def define_folder_structure(self):
        raise NotImplementedError

    def define_callbacks(self):
        return []

    def get_exp_root_folder(self):
        path = ROOT / self.name / self.exp_type
        return get_or_create_dir(path)

    def get_exp_data_root_folder(self):
        path = self.root_folder / "DATA"
        return get_exp_data_root_folder(path)

    def run(self):
        raise NotImplementedError


def ModelSelection(Experiment):
    exp_type = "MODEL_SELECTION"
    splitter_class = CVHoldoutSplitter


def ModelEvaluation(Experiment):
    exp_type = "MODEL_EVALUATION"
    splitter_class = HoldoutSplitter

