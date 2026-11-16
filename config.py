from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

VALIDATION = ConfigDict(
    extra="forbid", strict=True, str_min_length=1, allow_inf_nan=False
)


class TrainerConfig(BaseModel):
    model_config = VALIDATION
    model: str
    revision: str
    steps: PositiveInt
    batch_size: PositiveInt
    seed: int
    learning_rate: float = Field(gt=0)


class GeneratorConfig(BaseModel):
    model_config = VALIDATION
    sampling_params: dict[str, Any]
    engine: dict[str, Any]


class WandbConfig(BaseModel):
    model_config = VALIDATION
    mode: Literal["offline", "online", "disabled"]
    project: str


class Config(BaseModel):
    model_config = VALIDATION
    trainer: TrainerConfig
    generator: GeneratorConfig
    wandb: WandbConfig
    output_dir: str
