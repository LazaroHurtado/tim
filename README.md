The goal of this repo is to show Train-Inference Mismatch (TIM) between a trainer and generator during RL training.

This repro uses Ray for managing actors for the trainer and generator. Each requests 1 GPU on the local machine and uses NCCL for weight sync. Feel free to update `config.yaml` for your workload but by default we GRPO RL train Qwen2.5-0.5B-Instruct on GSM8K dataset.

The main metric to track on wandb is `mismatch/p50`, `mismatch/p90`, `mismatch/p99`, `mismatch/mean`, and `mismatch/max` which measures the p50, p90, p99, average and max difference between trainer and generator logprobs for the same sampled tokens and prefixes on all generated ids in the batch.

## How to run

You will need `uv`, Python 3.12, and two available CUDA GPUs:

```bash
uv sync
uv run wandb login
uv run python main.py
```

The default run trains for 1,000 steps and logs to W&B