"""Preparación de datasets para fine-tuning de DeepSeek-OCR."""

from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset

from src.config import Config


def convert_to_conversation(sample: dict[str, Any], instruction: str) -> dict[str, list[dict[str, Any]]]:
    """Convierte una muestra MolParser al formato conversacional de DeepSeek-OCR.

    Parameters
    ----------
    sample : dict[str, Any]
        Muestra con campos `image` y `text` (SMILES).
    instruction : str
        Prompt de usuario con token de imagen.

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        Estructura `messages` compatible con el collator de entrenamiento.
    """
    return {
        "messages": [
            {"role": "<|User|>", "content": instruction, "images": [sample["image"]]},
            {"role": "<|Assistant|>", "content": sample["text"]},
        ]
    }


def load_training_dataset(config: Config | None = None) -> list[dict[str, Any]]:
    """Carga y transforma el subset de entrenamiento desde Hugging Face.

    Parameters
    ----------
    config : Config | None
        Configuración del proyecto.

    Returns
    -------
    list[dict[str, Any]]
        Lista de conversaciones listas para el Trainer.
    """
    cfg = config or Config()
    instruction = cfg.inference.prompt

    raw = load_dataset(
        cfg.training.dataset_name,
        cfg.training.dataset_config,
        split=cfg.training.train_split,
    )
    raw = raw.rename_column("SMILES", "text")

    return [convert_to_conversation(sample, instruction) for sample in raw]


def load_eval_sample(config: Config | None = None, index: int = 15) -> Dataset:
    """Carga una muestra de evaluación no usada en entrenamiento."""
    cfg = config or Config()
    return load_dataset(
        cfg.training.dataset_name,
        cfg.training.dataset_config,
        split="train[:50]",
    )[index]
