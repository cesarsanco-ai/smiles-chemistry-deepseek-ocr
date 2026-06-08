"""Configuración centralizada del pipeline OCR químico."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LoRAConfig:
    """Hiperparámetros de adaptadores LoRA."""

    rank: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class InferenceConfig:
    """Parámetros de inferencia DeepSeek-OCR."""

    prompt: str = "<image>\nFree OCR. "
    base_size: int = 1024
    image_size: int = 640
    crop_mode: bool = True


@dataclass
class TrainingConfig:
    """Hiperparámetros de fine-tuning."""

    dataset_name: str = "UniParser/MolParser-7M"
    dataset_config: str = "test_simple_10k"
    train_split: str = "train[:25]"
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    max_steps: int = 10
    learning_rate: float = 2e-4
    warmup_steps: int = 2
    seed: int = 3407
    output_dir: str = "outputs"


@dataclass
class Config:
    """Configuración global del proyecto."""

    model_id: str = "unsloth/DeepSeek-OCR"
    local_model_dir: Path = Path("./deepseek_ocr")
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
