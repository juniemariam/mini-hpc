"""Fully Sharded Data Parallel training benchmark — same model/data/logging
as train_ddp.py, swapped to FSDP so you can directly compare memory
footprint and step time between the two sharding strategies.

    torchrun --nproc_per_node=2 train_fsdp.py --epochs 2 --batch-size 64
"""
import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from common import BenchmarkLogger, SimpleCNN, SyntheticImageDataset, timer


import platform


def setup_distributed():
    # NCCL is Linux-only — Windows PyTorch builds don't ship it, even with
    # CUDA available, so gloo is the correct backend choice on Windows.
    if platform.system() == "Windows":
        backend = "gloo"
    else:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # File-based rendezvous instead of the default TCPStore/libuv path — the
    # latter is broken on Windows for RTX 50-series (Blackwell) GPUs as of
    # this writing (see pytorch/pytorch#165959). FileStore does the same
    # job without touching that code path.
    init_file = os.environ.get("DDP_INIT_FILE", "C:/temp/ddp_sync")
    dist.init_process_group(
        backend=backend,
        init_method=f"file:///{init_file}",
        world_size=world_size,
        rank=rank,
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-samples", type=int, default=20000)
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--run-name", type=str, default="fsdp_run")
    args = parser.parse_args()

    # FSDP requires a real process group — this script assumes torchrun launch.
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    dataset = SyntheticImageDataset(num_samples=args.num_samples)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=2)

    model = SimpleCNN().to(device)
    model = FSDP(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    logger = BenchmarkLogger(args.log_dir, f"{args.run_name}_rank{rank}") if rank == 0 else None

    step = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with timer() as elapsed:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            step_time = elapsed()

            if step % 10 == 0:
                msg = f"[rank {rank}] epoch {epoch} step {step} loss {loss.item():.4f} step_time {step_time*1000:.1f}ms"
                print(msg, flush=True)
                if logger:
                    logger.log_step(step, loss.item(), step_time, extra={
                        "epoch": epoch,
                        "world_size": world_size,
                        "batch_size": args.batch_size,
                        "mode": "fsdp",
                    })
            step += 1

    if logger:
        logger.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
