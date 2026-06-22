"""Unified logging: TensorBoard, file log, and optional WandB."""

import os
import json
import time
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Unified logger for training metrics.

    Args:
        log_dir: root directory for logs
        algo_name: algorithm name (ippo/mappo/qmix)
        use_wandb: whether to use Weights & Biases
        wandb_project: WandB project name
        config: optional config dict to save
    """

    def __init__(
        self,
        log_dir: str,
        algo_name: str,
        use_wandb: bool = False,
        wandb_project: str = "quadrapong",
        config: Optional[dict] = None,
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(log_dir) / f"{algo_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.tb_writer = SummaryWriter(self.run_dir / "tensorboard")
        self.algo_name = algo_name

        # File log
        self.log_file = self.run_dir / "train.log"

        # WandB
        self.use_wandb = use_wandb
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=f"{algo_name}_{timestamp}",
                    config=config,
                    dir=str(self.run_dir),
                )
            except Exception as e:
                print(f"WandB init failed: {e}, falling back to TensorBoard only")
                self.use_wandb = False

        # Save config
        if config:
            with open(self.run_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)

        self._start_time = time.time()

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value to TensorBoard and optionally WandB."""
        full_tag = f"{self.algo_name}/{tag}"
        self.tb_writer.add_scalar(full_tag, value, step)
        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.log({full_tag: value}, step=step)

    def log_scalars(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """Log multiple scalars at once."""
        for k, v in metrics.items():
            tag = f"{prefix}/{k}" if prefix else k
            self.log_scalar(tag, v, step)

    def log_eval(self, metrics: Dict[str, Any], step: int):
        """Log evaluation metrics (handles non-scalar values gracefully)."""
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.log_scalar(f"eval/{k}", float(v), step)

    def print(self, msg: str, also_log: bool = True):
        """Print to stdout and optionally write to log file."""
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        if also_log:
            with open(self.log_file, "a") as f:
                f.write(line + "\n")

    def get_elapsed(self) -> float:
        """Return elapsed seconds since logger creation."""
        return time.time() - self._start_time

    def close(self):
        self.tb_writer.close()
        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.finish()


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    elif m > 0:
        return f"{m}m{s:02d}s"
    else:
        return f"{s}s"
