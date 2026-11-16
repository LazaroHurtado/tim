import re

import ray
import torch
from transformers import AutoModelForCausalLM
from vllm.distributed.weight_transfer.base import ModuleSource
from vllm.distributed.weight_transfer.clients import RayVLLMWeightSyncClient
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerInitInfo,
    NCCLTrainerWeightTransferEngine,
)
from vllm.utils.network_utils import get_open_port


def final_answer(text):
    numbers = re.findall(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)", text)
    return float(numbers[-1].replace(",", "")) if numbers else None


def advantages(rewards, group_size):
    groups = torch.tensor(rewards, dtype=torch.float32).reshape(-1, group_size)
    return (
        (groups - groups.mean(1, keepdim=True))
        / (groups.std(1, keepdim=True, unbiased=False) + 1e-4)
    ).flatten()


def token_logprobs(logits, tokens, temperature=1.0):
    return (
        (logits.float() / temperature)
        .log_softmax(-1)
        .gather(-1, tokens[:, None])
        .squeeze(-1)
    )


def get_rewards(rollouts):
    rewards = []
    for rollout in rollouts:
        golden = final_answer(rollout["answer"])
        pred = final_answer(rollout["text"])
        if golden is None:
            raise ValueError("GSM8K reference answer has no final number.")

        reward = 0
        if not rollout["truncated"] and pred is not None:
            # Provide learning signal when the answer is some what close, clipped to positive rewards only
            denom = abs(golden) if golden != 0.0 else 1.0
            reward = max(0, 1 - abs(pred - golden) / denom)
        rewards.append(reward)

    return rewards


@ray.remote(num_gpus=1, num_cpus=2)
class Trainer:
    def __init__(self, model_path, learning_rate):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.float32, attn_implementation="sdpa"
        ).to("cuda")
        self.model.eval()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=0.0
        )
        self.updates = 0
        self.transfer = None

    def connect(self, generator):
        self.transfer = NCCLTrainerWeightTransferEngine.trainer_init(
            NCCLTrainerInitInfo(
                master_address=ray.util.get_node_ip_address(),
                master_port=get_open_port(),
                world_size=2,
                rank=0,
                packed=False,
            ),
            client=RayVLLMWeightSyncClient(generator),
            source=ModuleSource(self.model),
        )

    def step(self, rollouts, group_size):
        if self.transfer is None:
            raise RuntimeError("Connect the trainer to the generator before training.")
        rewards = get_rewards(rollouts)
        group_advantages = advantages(rewards, group_size)

        self.optimizer.zero_grad(set_to_none=True)

        logprob_diff, scores, losses = [], [], []
        for rollout, advantage in zip(rollouts, group_advantages, strict=True):
            input_len = len(rollout["prompt"])
            output_len = len(rollout["tokens"])
            ids = torch.tensor(rollout["prompt"] + rollout["tokens"], device="cuda")

            with (
                torch.set_grad_enabled(advantage.item() != 0),
                torch.autocast("cuda", dtype=torch.bfloat16),
            ):
                logits = self.model(ids[None], use_cache=False).logits[
                    0, input_len - 1 : input_len + output_len - 1
                ]
                targets = ids[input_len : input_len + output_len]

                trainer_logprobs = token_logprobs(logits, targets)
                loss = -advantage.to("cuda") * trainer_logprobs.mean() / len(rollouts)

            if advantage.item() != 0:
                loss.backward()

            with torch.no_grad():
                generator_logprobs = torch.tensor(rollout["logprobs"], device="cuda")
                logprob_diff.append(
                    (trainer_logprobs.detach() - generator_logprobs).abs().cpu()
                )

                scores.append(trainer_logprobs.detach().cpu())
                losses.append(loss.item())

            del logits, trainer_logprobs, generator_logprobs, loss

        updated = bool(group_advantages.any())
        grad_norm = 0.0
        if updated:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 1.0, error_if_nonfinite=True
            ).item()

            self.optimizer.step()
            self.updates += 1

        logprob_diff = torch.cat(logprob_diff)
        metrics = self.compute_metrics(
            rollouts,
            rewards,
            loss=sum(losses),
            grad_norm=grad_norm,
            updates=self.updates,
            updated=int(updated),
            **{
                "logprob/trainer": torch.cat(scores).mean().item(),
                "mismatch/mean": logprob_diff.mean().item(),
                "mismatch/max": logprob_diff.max().item(),
            },
        )

        if updated:
            with torch.no_grad():
                self.transfer.send_weights()

        return metrics

    def close(self):
        if self.transfer is not None:
            self.transfer.shutdown()

    def compute_metrics(self, rollouts, rewards, **kwargs):
        metrics = {"reward": sum(rewards) / len(rewards), **kwargs}

        metrics["completion/truncated_fraction"] = sum(
            rollout["truncated"] for rollout in rollouts
        ) / len(rollouts)

        metrics["completion/mean_length"] = sum(
            len(rollout["tokens"]) for rollout in rollouts
        ) / len(rollouts)

        metrics["logprob/generator"] = sum(
            sum(rollout["logprobs"]) for rollout in rollouts
        ) / sum(len(rollout["tokens"]) for rollout in rollouts)

        return metrics
