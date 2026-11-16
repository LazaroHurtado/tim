import math

import ray
from vllm import LLM, SamplingParams


@ray.remote(num_gpus=1, num_cpus=2)
class Generator:
    def __init__(self, model_path, config):
        self.config = config

        self.sampling_params = SamplingParams(**config.sampling_params)
        self.model = LLM(model=model_path, **config.engine)

    def generate(self, batch, seed):
        sampling_params = self.sampling_params.clone()
        sampling_params.seed = seed
        outputs = self.model.generate(
            [{"prompt_token_ids": row["prompt"]} for row in batch],
            sampling_params,
            use_tqdm=False,
        )

        rollouts = []
        for row, output in zip(batch, outputs, strict=True):
            if output.prompt_token_ids != row["prompt"]:
                raise ValueError("Generator changed the prompt token IDs.")
            if len(output.outputs) != self.config.sampling_params["n"]:
                raise ValueError("Generator returned an incomplete rollout group.")

            for completion in output.outputs:
                if not completion.token_ids or completion.logprobs is None:
                    raise ValueError("Generator returned no tokens or logprobs.")

                # collect logprobs of vocabulary tokens for every generated token in the completion
                scores = []
                for token, logprobs in zip(
                    completion.token_ids, completion.logprobs, strict=True
                ):
                    token_logprob = logprobs[token].logprob
                    if not math.isfinite(token_logprob):
                        raise ValueError("Generator returned nonfinite logprobs.")

                    scores.append(token_logprob)

                rollouts.append(
                    {
                        "prompt": row["prompt"],
                        "tokens": list(completion.token_ids),
                        "logprobs": scores,
                        "text": completion.text,
                        "answer": row["answer"],
                        "truncated": completion.finish_reason == "length",
                    }
                )

        return rollouts

    def init_weight_transfer_engine(self, request):
        self.model.init_weight_transfer_engine(request)

    def start_weight_update(self):
        self.model.start_weight_update()

    def update_weights(self, request):
        self.model.update_weights(request)

    def finish_weight_update(self):
        self.model.finish_weight_update()
        if self.config.engine.get("enable_prefix_caching"):
            if not self.model.reset_prefix_cache():
                raise RuntimeError("vLLM did not clear its stale prefix cache.")

    def shutdown_weight_transfer_engine(self):
        self.model.shutdown_weight_transfer_engine()

    def close(self):
        self.shutdown_weight_transfer_engine()
        self.model.llm_engine.engine_core.shutdown()
