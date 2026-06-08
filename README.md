# OCR de Estructuras Químicas → SMILES

Solución de **visión por computadora** para extraer notación **SMILES** desde imágenes de moléculas, basada en **DeepSeek-OCR** con fine-tuning **LoRA**.

| | |
|---|---|
| **Autor** | Carlos Cesar Sanchez Coronel |
| **Modelo base** | [unsloth/DeepSeek-OCR](https://huggingface.co/unsloth/DeepSeek-OCR) (3B) |
| **Dataset entrenamiento** | [UniParser/MolParser-7M](https://huggingface.co/datasets/UniParser/MolParser-7M) |
| **Demo** | [cesarsanco-ai.github.io/SMILES-CHEMISTRY-DEEPSEEK-OCR](https://cesarsanco-ai.github.io/SMILES-CHEMISTRY-DEEPSEEK-OCR/) |

---

## Descripción

El proyecto convierte diagramas 2D de estructuras químicas en cadenas **SMILES** mediante un pipeline de OCR multimodal. Utiliza la compresión óptica de contexto de DeepSeek-OCR y adapta el modelo con LoRA para el dominio químico.

Complementariamente incluye un **generador sintético de moléculas** (RDKit) con 36 compuestos en estilos clean y realistic, útil para pruebas y augmentación visual.

---

## Enfoque técnico

- **DeepSeek-OCR (3B)** con DeepEncoder (SAM + CLIP + compresor 16×)
- **Fine-tuning LoRA** (rank 16) — 2.27% de parámetros entrenables
- **Resolución Gundam:** base 1024 px, crop 640 px
- **Dataset:** subset de MolParser-7M (25 muestras de entrenamiento)
- **Salida:** SMILES en formato `<smiles>...</smiles>`

---

## Configuración de entrenamiento

| Parámetro | Valor |
|-----------|-------|
| Parámetros entrenables | 77.5M / 3.4B |
| LoRA rank / alpha | 16 / 16 |
| Steps | 10 |
| Batch efectivo | 4 |
| Learning rate | 2e-4 |
| Optimizador | AdamW 8-bit |
| GPU | Tesla T4 (Colab) |

Detalle en [`docs/metrics.json`](docs/metrics.json) y [`docs/pipeline.md`](docs/pipeline.md).

---

## Arquitectura

```text
Imagen molecular (PNG)
        │
        ▼
DeepEncoder → tokens visuales comprimidos
        │
        ▼
DeepSeek-OCR + LoRA
        │
        ▼
<smiles> CC(=O)Oc1ccccc1C(=O)O </smiles>
```

---

## Estructura del repositorio

```text
SMILES-CHEMISTRY-DEEPSEEK-OCR/
├── index.html                 # Presentación del proyecto
├── assets/                    # Estilos e imágenes de muestra
├── src/                       # Código modular
│   ├── config.py
│   ├── dataset.py
│   ├── inference.py
│   └── training.py
├── notebooks/
│   └── notebook_colab.ipynb
└── docs/
    ├── pipeline.md
    ├── metrics.json
    └── dataset_catalog.json
```

---

## Stack tecnológico

Python 3.11 · PyTorch · Transformers · Unsloth · PEFT · RDKit · Hugging Face Datasets

---

## Dataset

El dataset sintético de 36 moléculas (RDKit) y el modelo base (~3B parámetros) no se incluyen en el repositorio. El catálogo se documenta en [`docs/dataset_catalog.json`](docs/dataset_catalog.json). Las imágenes en `assets/` son muestras representativas. El entrenamiento utiliza MolParser-7M desde Hugging Face en Google Colab.

---

## Referencias

- [DeepSeek-OCR Paper (arXiv:2510.18234)](https://arxiv.org/abs/2510.18234)
- [DeepSeek-OCR GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)
- [MolParser-7M Dataset](https://huggingface.co/datasets/UniParser/MolParser-7M)

---

## Contacto

**Carlos Cesar Sanchez Coronel**  
[github.com/cesarsanco-ai](https://github.com/cesarsanco-ai)
