The goal of this repo is to show Train-Inference Mismatch (TIM) between a trainer and generator during RL training.

This repro uses Ray for managing actors for the trainer and generator. Each requests 1 GPU on the local machine and uses NCCL for weight sync. Feel free to update `config.yaml` for your workload but by default we GRPO RL train Qwen2.5-0.5B-Instruct on GSM8K dataset.

The main metric to track on wandb is `mismatch/mean` and `mismatch/max` which measures the average and max difference between trainer and generator logprobs for the same sampled tokens and prefixes on all generated ids in the batch.