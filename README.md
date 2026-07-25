# Mini-HPC: GPU Cluster Orchestration & Distributed Training Benchmark Suite

A small, honest recreation of the tradeoffs an ML/HPC infra engineer deals with daily:
same training job, run under different schedulers (Kubernetes vs. Slurm), different
container runtimes (Docker vs. Enroot), and different storage/networking backends
(EBS vs. FSx for Lustre, standard vs. EFA-enabled networking) — with real
measurements at each step, not just claims.

## Why this project exists

Built to demonstrate hands-on understanding of the infra stack behind large-scale
GPU training clusters: scheduling & orchestration, containers, storage, networking,
and distributed training frameworks — the exact surface area of an AI/ML
Infrastructure Engineer role supporting GPU research clusters.

## Phases

1. **Phase 1 (this scaffold): DDP/FSDP training + Docker, local single-GPU.**
   Runs on any single CUDA GPU (developed on an RTX 5070 Ti). Establishes the
   baseline training script and benchmark instrumentation before anything touches
   a cluster.
2. **Phase 2: Kubernetes Job vs. Slurm sbatch.** Same container image, two
   schedulers, on a short-lived 2-3 node cloud burst (AWS spot GPU instances).
   Compare cold-start time, queueing behavior, and failure handling.
3. **Phase 3: Docker vs. Enroot.** Same image, unprivileged HPC-native runtime vs.
   standard Docker — compare startup time and isolation model.
4. **Phase 4: Storage.** FSx for Lustre vs. EBS for data-loading throughput during
   training.
5. **Phase 5 (stretch): Networking + framework comparison.** EFA vs. non-EFA
   all-reduce step time; PyTorch DDP/FSDP vs. JAX pmap vs. NeMo, memory and
   scaling efficiency.

## What's real vs. simulated (say this out loud in interviews)

- Phases 1–4 use real cloud resources at small, short-lived scale — genuine
  measurements, not fabricated numbers.
- True Infiniband and IBM Spectrum LSF are not accessible to an individual;
  Phase 5's EFA comparison is the closest honest proxy for "why does
  high-speed networking matter for distributed training," and LSF is discussed
  conceptually only, never claimed as hands-on.
- GPFS/BeeGFS are documented as compare/contrast notes against Lustre rather
  than stood up — not worth the cost/complexity for a portfolio project.

## Repo layout

```
mini-hpc/
  training/       # DDP + FSDP training scripts, shared benchmark utilities
  docker/         # Container image definition + Enroot import notes
  k8s/            # Kubernetes Job manifests
  slurm/          # Slurm sbatch scripts
  scripts/        # GPU monitoring + benchmark report tooling
  infra/          # Setup notes for FSx/Lustre, EFA, cost controls
```

## Quickstart (Phase 1, local)

```bash
cd training
pip install -r ../requirements.txt

# Single GPU
python train_ddp.py --epochs 2 --batch-size 64

# Multi-GPU on one machine (if you have >1 GPU)
torchrun --nproc_per_node=2 train_ddp.py --epochs 2 --batch-size 64

# FSDP variant
torchrun --nproc_per_node=2 train_fsdp.py --epochs 2 --batch-size 64
```

Benchmark logs are written to `./logs/` as JSONL — one line per step with
timing and (if available) GPU utilization. Turn them into a report with:

```bash
python ../scripts/bench_report.py logs/*.jsonl
```

## Next steps once Phase 1 runs cleanly

See `infra/NOTES.md` for the AWS setup for Phase 2 (spot GPU instances,
Kubernetes vs. Slurm) and Phase 4 (FSx for Lustre).
