# Pipeline técnico — OCR de estructuras químicas (SMILES)

Documentación de la solución basada en **DeepSeek-OCR** para extraer notación SMILES desde imágenes de moléculas.

## 1. Problema

Las estructuras químicas en papers, patentes y bases de datos aparecen frecuentemente como **diagramas 2D**. Convertirlas a **SMILES** (Simplified Molecular Input Line Entry System) permite búsqueda, almacenamiento y procesamiento computacional en quimioinformática.

## 2. Enfoque

| Aspecto | Detalle |
|---------|---------|
| Modelo base | DeepSeek-OCR (3B) — compresión óptica de contexto |
| Adaptación | LoRA (rank 16) sobre proyecciones de atención y MLP |
| Entrada | Imagen PNG/JPG de estructura molecular |
| Salida | Cadena SMILES dentro de etiquetas `<smiles>` |
| Entorno | Google Colab + Unsloth |

## 3. Flujo del pipeline

```text
Imagen molecular (PNG)
        │
        ▼
DeepEncoder (SAM + CLIP + compresor 16×)
        │
        ▼
Tokens visuales comprimidos
        │
        ▼
DeepSeek-OCR (3B) + LoRA
        │
        ▼
Texto SMILES: <smiles>...</smiles>
```

## 4. Fine-tuning con LoRA

- **Dataset de entrenamiento:** `UniParser/MolParser-7M` (subset de 25 muestras).
- **Formato:** conversación usuario/asistente con imagen embebida.
- **Parámetros entrenables:** 77.5M de 3.4B (2.27%).
- **Configuración:** 10 steps, batch efectivo 4, lr=2e-4, AdamW 8-bit.
- **Resolución Gundam:** base 1024 px, crop 640 px.

## 5. Generación de datos sintéticos

Complementariamente se desarrolló un generador con **RDKit** que produce 36 moléculas representativas en dos estilos:

- **Clean:** diagramas limpios tipo textbook.
- **Realistic:** variaciones visuales que simulan escaneos o capturas.

Categorías: alcoholes, ácidos carboxílicos, aromáticos, heterociclos, aminoácidos, fármacos, azúcares, cicloalcanos, aldehídos, cetonas y aminas.

## 6. Arquitectura DeepSeek-OCR

- **SAM (80M):** percepción local con window attention.
- **CLIP (300M):** conocimiento semántico con global attention.
- **Compresor convolucional 16×:** reduce tokens visuales hasta 10× respecto al texto equivalente.

## 7. Entorno de ejecución

La solución fue desarrollada en **Google Colab** con GPU Tesla T4. El modelo base se obtiene desde Hugging Face (`unsloth/DeepSeek-OCR`). El código modular en `src/` organiza configuración, preparación de datos e inferencia.

## 8. Referencias

- Paper: [DeepSeek-OCR arXiv:2510.18234](https://arxiv.org/abs/2510.18234)
- Modelo: [unsloth/DeepSeek-OCR](https://huggingface.co/unsloth/DeepSeek-OCR)
- Dataset: [UniParser/MolParser-7M](https://huggingface.co/datasets/UniParser/MolParser-7M)
