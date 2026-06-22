# A Study on AI-generated dermatology clinical vignettes

Python pipeline for extracting, processing, and analyzing AI-generated dermatology clinical vignettes. Supports text embedding, t-SNE visualization, and comparison of vignettes generated with vs. without dermoscopy images across six skin conditions.

## Overview

This repository processes supplementary vignette data from a JAAD-series study comparing large multimodal model (LMM) outputs on dermatology case challenges. Vignettes are parsed from `.docx` files, embedded via OpenAI or MedGemma models, and visualized using t-SNE to assess semantic clustering by diagnosis and image-availability condition.

**Six skin conditions covered:**
- Melanoma
- Nevus
- Basal Cell Carcinoma
- Dermatofibroma
- Seborrheic Keratosis
- Squamous Cell Carcinoma

---

## Repository Structure

| File | Description |
|---|---|
| `extract_vignettes_txt_only.py` | Parses text-only vignettes from structured `.docx` tables → JSON |
| `extract_vignettes_txt_only_hl.py` | Variant with highlighted/annotated text extraction |
| `extract_vignettes_w_img.py` | Parses vignettes generated with accompanying dermoscopy images → JSON |
| `tsne_all_6_conditions.py` | t-SNE of all 6 diagnoses (with-image vs. without-image) using OpenAI embeddings |
| `tsne_all_6_conditions_plot_caseid.py` | Same t-SNE but annotated with case IDs |
| `txt_embed_plots_medgemma.py` | Embedding + visualization pipeline using MedGemma |
| `txt_embed_plots_openai.py` | Embedding + visualization pipeline using OpenAI `text-embedding-3-small` |
| `txt_embed_plots_openai_mel_ben_all.py` | OpenAI embedding for melanoma + benign, all conditions |
| `txt_embed_plots_openai_melben.py` | OpenAI embedding restricted to melanoma vs. benign subset |
| `txt_embed_plots_wo_img.py` | Embedding pipeline for text-only (without image) vignettes |

---

## Pipeline

```
.docx (Supplementary Text)
        │
        ▼
extract_vignettes_*.py  →  JSON [{case_number, diagnosis, vignette}, ...]
        │
        ▼
txt_embed_plots_*.py    →  OpenAI / MedGemma embeddings
        │
        ▼
tsne_*.py               →  t-SNE plots (per-diagnosis + combined, PNG, 300 DPI)
```

---

## Dependencies

```bash
pip install openai python-docx pandas scikit-learn matplotlib numpy
```

For MedGemma embeddings, a local or API-accessible MedGemma model is required.

---

## Configuration

Input paths and output directories are currently hardcoded to local paths (`/mnt/d/Naved/Data/derm_vignette/`). Update these before running:

```python
file_path = "path/to/your/Supplementary_Text.docx"
output_file = "path/to/output.json"
output_dir = "path/to/output_figures/"
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="sk-..."
```

---

## Usage

**Step 1 — Extract vignettes:**
```bash
python extract_vignettes_txt_only.py      # text-only condition
python extract_vignettes_w_img.py         # with-image condition
```

**Step 2 — Embed and visualize:**
```bash
python tsne_all_6_conditions.py           # combined t-SNE, all 6 conditions
python txt_embed_plots_openai.py          # per-condition embedding plots
```

---

## Output

- `tsne_all_6_conditions_<timestamp>.png` — Combined scatter plot: 6 diagnoses × 2 conditions (filled = with image, open = without image), distinct marker shapes per diagnosis
- Per-diagnosis t-SNE PNGs: `tsne_melanoma_<timestamp>.png`, etc.
- Intermediate JSONs with structured vignette records

---

## Citation

If you use this pipeline, please cite the associated manuscript (in preparation / under review).

---

## Author

[Mohammad Iqbal Nouyed (Naved)](https://github.com/iqbalnaved)  
Independent Researcher, West Virginia University — Department of Microbiology, Immunology and Cell Biology  
PI: Gangqing Hu | Collaborator: Donald Adjeroh
