DEFAULT_EXP_DIR = "EXPERIMENTS"
DEFAULT_DATA_DIR = "DATA"

TRAINING = "training"
VALIDATION = "validation"
TEST = "test"
LEARNING_MODES = [TRAINING, VALIDATION, TEST]

DEFAULTS = {
    "max_epochs": 10,
    "device": "cpu",
    "model": {"class_name": "modules.models.MLP", "params": {"dim_layers": [128, 64]}},
    "criterion": {"class_name": "modules.criterions.BCE", "params": {}},
    "data": {
        "processor": {
            "root": "DATA",
            "name": "toy_classification",
            "raw_dir_name": "raw",
            "processed_dir_name": "processed",
            "dataset_filename": "dataset.pt",
            "splitter": {
                "splits_dir_name": "splits",
                "splits_filename": "splits.yaml",
                "class_name": "data.splitters.HoldoutSplitter",
                "params": {},
            },
        },
        "provider": {
            "loader": {
                "class_name": "torch.utils.data.DataLoader",
                "params": {"batch_size": 32, "shuffle": True},
            }
        },
    },
    "optimizer": {"class_name": "torch.optim.Adam", "params": {}},
    "callbacks": {},
}

DEFAULT_LAST_CKPT_FILENAME = 'last.pt'
DEFAULT_BEST_CKPT_FILENAME = 'best.pt'

DEFAULT_CSV_TRAINING_FILENAME = 'training.csv'
DEFAULT_CSV_TEST_FILENAME = 'test.log'
