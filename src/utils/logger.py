import logging
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any
import json
from pathlib import Path

class Logger:

    def __init__(
        self,
        name: str,
        log_dir: str = "logs",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        self.logger.handlers = []

        self.logger.propagate = False

        os.makedirs(log_dir, exist_ok=True)

        console_formatter = logging.Formatter(
            '%(message)s'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        today = datetime.now().strftime('%Y-%m-%d')
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f'{today}.log')
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        self.metrics: Dict[str, Any] = {}
        self.results_dir = "results"

    @property
    def handlers(self):
        return self.logger.handlers

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def critical(self, msg: str) -> None:
        self.logger.critical(msg)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        self.metrics.update(metrics)

        metrics_str = " - ".join([f"{k}: {v}" for k, v in metrics.items()])
        if step is not None:
            metrics_str = f"Step {step} - {metrics_str}"

        self.info(metrics_str)

    def save_metrics(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        self.info(f"Saved metrics to {filepath}")

def get_logger(
    name: str = __name__,
    log_dir: str = "logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> Logger:
    return Logger(name, log_dir, console_level, file_level)
