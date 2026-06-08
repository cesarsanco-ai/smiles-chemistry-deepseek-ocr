"""Inferencia con DeepSeek-OCR para extracción de SMILES."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.config import Config

logger = logging.getLogger(__name__)


def run_inference(
    model: Any,
    tokenizer: Any,
    image_path: str | Path,
    output_path: str | Path = "outputs",
    config: Config | None = None,
) -> Any:
    """Ejecuta inferencia OCR sobre una imagen molecular.

    Parameters
    ----------
    model : Any
        Modelo DeepSeek-OCR cargado (Unsloth FastVisionModel).
    tokenizer : Any
        Tokenizador asociado al modelo.
    image_path : str | Path
        Ruta a la imagen de la estructura química.
    output_path : str | Path
        Directorio de salida para artefactos de inferencia.
    config : Config | None
        Configuración de resolución y prompt.

    Returns
    -------
    Any
        Resultado devuelto por `model.infer`.
    """
    cfg = config or Config()
    inf = cfg.inference

    logger.info("Inferencia sobre %s", image_path)
    return model.infer(
        tokenizer,
        prompt=inf.prompt,
        image_file=str(image_path),
        output_path=str(output_path),
        base_size=inf.base_size,
        image_size=inf.image_size,
        crop_mode=inf.crop_mode,
        save_results=True,
        test_compress=False,
    )


def extract_smiles_tag(raw_output: str) -> str | None:
    """Extrae el contenido SMILES de la salida del modelo."""
    start = raw_output.find("<smiles>")
    end = raw_output.find("</smiles>")
    if start == -1 or end == -1:
        return None
    return raw_output[start + len("<smiles>") : end].strip()
