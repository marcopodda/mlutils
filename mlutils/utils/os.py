import os
from pathlib import Path


def maybe_makedir(path):
    path = Path(path)
    if not path.exists():
        os.makedirs(path)
    return path