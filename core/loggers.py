import pandas as pd
from pathlib import Path

from .events import EventHandler


class CSVLogger(EventHandler):
    def __init__(self, logdir=Path('logs')):
        self.filename = logdir / "results.log"

    def on_epoch_end(self, state):
        df = pd.DataFrame(state.results).round(6)
        df.to_csv(self.filename, index=False)

