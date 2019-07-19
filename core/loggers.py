import pandas as pd
from pathlib import Path

from .events import EventHandler


class CSVLogger(EventHandler):
    def __init__(self, logdir=Path('logs')):
        self.logdir = logdir

    def on_epoch_end(self, state):
        filename = self.logdir / "results.log"
        if not filename.exists() or state.epoch == 0:
            df = pd.DataFrame([state.epoch_results])
        else:
            df = pd.read_csv(filename, index_col=False)
            df = pd.concat([df, pd.DataFrame([state.epoch_results])], sort=False)

        df.round(6).to_csv(filename, index=False)

    def on_test_epoch_end(self, state):
        results = {k: v for (k,v) in state.epoch_results.items() if 'test' in k}
        df = pd.DataFrame([results])
        df.round(6).to_csv(self.logdir / "test.log", index=False)

