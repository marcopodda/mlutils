import pandas as pd
from pathlib import Path


class CSVLogger:
    def __init__(self, monitor_on=['training', 'validation', 'test'], log_every=10, logdir=Path('logs')):
        self.logdir = logdir
        self.log_every = 10
        self.monitor_on = monitor_on

        self.filenames = {phase: self.logdir / f"{phase}.log" for phase in monitor_on}
        self.buffers = {}

    def init(self, metrics):
        for phase in self.monitor_on:
            if self.filenames[phase].exists():
                self.buffers[phase] = pd.read_csv(self.filenames[phase], index_col=False)
            else:
                columns = []
                for metric in metrics:
                    columns.append(metric.name)
                self.buffers[phase] = pd.DataFrame(columns=columns)

    def log(self, phase, values):
        if phase in self.monitor_on:
            values = pd.DataFrame(values, index=[0])
            self.buffers[phase] = pd.concat([self.buffers[phase], values], sort=False)
            if len(self.buffers[phase]) == self.log_every:
                self._empty_buffer(phase)

    def _empty_buffer(self, phase):
        if phase in self.monitor_on:
            self.buffers[phase].to_csv(self.filenames[phase], index=False)

