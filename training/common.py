"""Shared model, data, and benchmark-logging utilities for the mini-hpc
training scripts. Kept deliberately small (no big dataset downloads) so the
project runs anywhere, including on a laptop, and the interesting part is
the *infra* around training, not the model itself.
"""
import json
import os
import subprocess
import time
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class SyntheticImageDataset(Dataset):
    """Synthetic image-classification dataset. Avoids any download/dataset
    dependency so the benchmark harness is fully self-contained — the point
    of this project is measuring infra behavior, not model quality.
    """

    def __init__(self, num_samples=20000, image_size=64, num_classes=10):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(idx)
        x = torch.randn(3, self.image_size, self.image_size, generator=g)
        y = torch.randint(0, self.num_classes, (1,), generator=g).item()
        return x, y


class SimpleCNN(nn.Module):
    """Small CNN, big enough to be GPU-bound at reasonable batch sizes,
    small enough to iterate on quickly."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512), nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def gpu_utilization():
    """Best-effort GPU utilization + memory read via nvidia-smi. Returns
    None if nvidia-smi isn't available (e.g. CPU-only dev box)."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=2,
        ).decode().strip().splitlines()[0]
        util, mem_used, mem_total = [p.strip() for p in out.split(",")]
        return {
            "gpu_util_pct": float(util),
            "mem_used_mib": float(mem_used),
            "mem_total_mib": float(mem_total),
        }
    except Exception:
        return None


class BenchmarkLogger:
    """Writes one JSON line per training step: step time, loss, GPU util.
    This is the instrumentation layer that turns "I ran a training job" into
    "here's the throughput and utilization data to back it up."
    """

    def __init__(self, log_dir, run_name):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, f"{run_name}.jsonl")
        self._f = open(self.path, "a")

    def log_step(self, step, loss, step_time_s, extra=None):
        record = {
            "step": step,
            "loss": float(loss),
            "step_time_s": step_time_s,
            "throughput_img_s": None,  # filled in by caller if batch size known
            "gpu": gpu_utilization(),
            "timestamp": time.time(),
        }
        if extra:
            record.update(extra)
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


@contextmanager
def timer():
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start
