import json
import os
import sys
import time
from pathlib import Path

import hydra
import wandb
from datasets import load_dataset
from huggingface_hub import snapshot_download
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import Config


def build_dataset(dataset, tokenizer, seed):
    return [
        {
            "prompt": tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": row["question"]
                        + "\nSolve step by step. End with #### <number>.",
                    }
                ],
                add_generation_prompt=True,
                return_dict=False,
            ),
            "answer": row["answer"],
        }
        for row in dataset.shuffle(seed=seed)
    ]


def train(trainer, generator, dataloader, config, wandb_run):
    import ray

    step = 0
    seed = config.generator.sampling_params["seed"]
    group_size = config.generator.sampling_params["n"]

    while step < config.trainer.steps:
        for batch in dataloader:
            step += 1
            start = time.perf_counter()

            rollouts = generator.generate.remote(batch, seed + step)
            trainer_metrics = trainer.step.remote(rollouts, group_size)

            metrics = {
                "step": step,
                **ray.get(trainer_metrics),
                "step_seconds": time.perf_counter() - start,
            }
            wandb_run.log(metrics, step=step)
            if step == config.trainer.steps:
                break

    ray.get([trainer.close.remote(), generator.close.remote()])


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    # Ray's uv hook otherwise recreates the CUDA environment inside its temp directory.
    os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    import ray

    from generator import Generator
    from trainer import Trainer

    config = Config.model_validate(OmegaConf.to_container(cfg, resolve=True))
    directory = Path(config.output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    model_path = snapshot_download(
        config.trainer.model, revision=config.trainer.revision
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = build_dataset(dataset, tokenizer, config.trainer.seed)
    dataloader = DataLoader(
        dataset, batch_size=config.trainer.batch_size, collate_fn=list
    )

    config.generator.engine.setdefault(
        "max_model_len",
        max(len(row["prompt"]) for row in dataset)
        + config.generator.sampling_params["max_tokens"],
    )
    config.generator.engine.setdefault(
        "max_num_seqs",
        config.trainer.batch_size * config.generator.sampling_params["n"],
    )
    settings = {**config.model_dump(), "resolved_revision": Path(model_path).name}
    with (directory / "config.json").open("x") as stream:
        json.dump(settings, stream, indent=2)
        stream.write("\n")

    ray.init(
        address="local",
        include_dashboard=False,
        _temp_dir=str(Path(__file__).resolve().parent / ".ray"),
        runtime_env={
            "py_executable": sys.executable,
            "working_dir": str(Path(__file__).resolve().parent),
            "excludes": [".venv", ".ray", "outputs", ".git"],
            "env_vars": {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
        },
    )

    try:
        if ray.cluster_resources().get("GPU", 0) < 2:
            raise RuntimeError("The trainer and generator need one CUDA GPU each.")

        trainer = Trainer.remote(model_path, config.trainer.learning_rate)
        generator = Generator.remote(model_path, config.generator)

        # Set up NCCL weight transfer connection
        ray.get(trainer.connect.remote(generator), timeout=120)

        with (
            wandb.init(
                project=config.wandb.project,
                mode=config.wandb.mode,
                dir=str(directory),
                config=settings,
                save_code=False,
                settings=wandb.Settings(disable_git=True),
            ) as run,
        ):
            train(trainer, generator, dataloader, config, run)

    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
