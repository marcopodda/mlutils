EXP_DIR = "EXPERIMENTS"
EXP_NAME = "EXPERIMENT"
DATA_DIR = "DATA"
CKPTS_DIR = "ckpts"
EXPDATA_DIR = "data"
RESULTS_DIR = "results"
CONFIG_DIR = "config"
LOGS_DIR = "logs"

TRAINING = "training"
VALIDATION = "validation"
TEST = "test"
LEARNING_MODES = [TRAINING, VALIDATION, TEST]

RAW_DIR = "raw"
PROCESSED_DIR = "processed"
SPLITS_DIR = "splits"
SPLITS_FILENAME = "splits.yaml"

DATASET_FILENAME = "dataset.pt"

LAST_CKPT_FILENAME = 'last.pt'
BEST_CKPT_FILENAME = 'best.pt'

CSV_TRAINING_FILENAME = 'training.csv'
CSV_TEST_FILENAME = 'test.log'

FEATURES_NAME = "x"
TARGET_NAME = "y"

CONFIG = {
    "engine": {
        "num_epochs": 10,
        "device": "cpu",
        "model": {
            "params": {
                "dim_layers": [128, 64]
            }
        },
        "criterion": {
            "params": {}
        },
        "optimizer": {
            'class_name': 'torch.optim.Adam',
            'params': {}
        },
        "callbacks": {},
    },
    "data": {
        "processor": {
            "dataset_name": "toy_binary_classification",
            "splitter": {
                "params": {}
            },
            "params": {
                "n_samples": 1000,
                "n_classes": 2,
                "n_features": 16
            }
        },
        "dataset": {

        },
        "provider": {
            "loader": {
                "params": {
                    "batch_size": 32,
                    "shuffle": True
                }
            }
        },
    },
}

SPLITTER = 'mlutils.data.splitter.HoldoutSplitter'
LOADER = 'torch.utils.data.DataLoader'

# self.processor_class = processor_class
# self.splitter_class = splitter_class
# self.provider_class = provider_class
# self.dataset_class = dataset_class
# self.loader_class = loader_class
# self.engine_class = engine_class
# self.model_class = model_class
# self.criterion_class = criterion_class