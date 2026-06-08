"""Utilidades de entrenamiento LoRA para DeepSeek-OCR."""

from __future__ import annotations

import logging
from typing import Any

from transformers import TrainingArguments
from unsloth import FastVisionModel, is_bf16_supported

from src.config import Config

logger = logging.getLogger(__name__)


def apply_lora(model: Any, config: Config | None = None) -> Any:
    """Aplica adaptadores LoRA al modelo de visión."""
    cfg = config or Config()
    lora = cfg.lora

    adapted = FastVisionModel.get_peft_model(
        model,
        target_modules=lora.target_modules,
        r=lora.rank,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        bias="none",
        random_state=cfg.training.seed,
        use_rslora=False,
        loftq_config=None,
    )
    logger.info("LoRA aplicado | rank=%s | alpha=%s", lora.rank, lora.alpha)
    return adapted


def build_training_arguments(config: Config | None = None) -> TrainingArguments:
    """Construye argumentos de entrenamiento para Hugging Face Trainer."""
    cfg = config or Config()
    tr = cfg.training

    return TrainingArguments(
        per_device_train_batch_size=tr.per_device_batch_size,
        gradient_accumulation_steps=tr.gradient_accumulation_steps,
        warmup_steps=tr.warmup_steps,
        max_steps=tr.max_steps,
        learning_rate=tr.learning_rate,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=tr.seed,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        output_dir=tr.output_dir,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )
