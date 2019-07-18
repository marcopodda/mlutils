from core.logger import Logger
from pathlib import Path


l = Logger(Path("."))
l.log_event("ciao")
l.log_value(1.0)
