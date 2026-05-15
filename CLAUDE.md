---
name: Project Overview – DINO ViT‑S16 Unsupervised Clustering
description: Recommended directory layout and brief description of each folder for a DINO‑based clustering workflow.
type: project
---

# DINO ViT‑S16 Unsupervised Clustering – Folder Structure

This file documents the recommended layout for a project that uses the **DINO ViT‑S16** model to extract features from images, reduce dimensionality, cluster the data, and evaluate the results.

## Top‑level layout

```
├── CLAUDE.md                # ← This documentation file
├── data/                    # Raw & generated data (git‑ignore)
│   ├── raw/                 # Original image files (not version‑controlled)
│   ├── processed/           # Pre‑processed tensors / cached images
│   └── manifest.csv         # Optional list of files + meta (e.g. labels)
├── src/                     # Source code (importable as a package)
│   ├── __init__.py
│   ├── data_loader.py       # Loads raw data, applies preprocessing, writes to data/processed/
│   ├── model.py             # DINO ViT‑S16 loading / optional fine‑tuning
│   ├── feature_extractor.py # Runs the model and saves raw features
│   ├── dim_reducer.py       # PCA / UMAP / t‑SNE reduction
│   ├── clustering.py        # K‑Means, DBSCAN, etc.
│   └── utils.py             # Helper functions (logging, eval metrics, etc.)
├── features/                # Raw DINO feature vectors (git‑ignore)
├── embeddings/              # Reduced‑dimensional embeddings (git‑ignore)
├── clusters/                # Cluster assignment files (git‑ignore)
├── plots/                   # PNG / HTML visualisations (git‑ignore)
├── results/                 # Human‑readable outcomes
│   ├── summary.md           # Observations, conclusions, next steps
│   └── cluster_assignments.csv
├── notebooks/               # Optional Jupyter notebooks for ad‑hoc analysis
│   └── explore.ipynb
├── requirements.txt         # Python dependencies (torch, torchvision, scikit‑learn, umap‑learn, etc.)
└── .gitignore               # Should ignore data/, features/, embeddings/, clusters/, plots/
```

## Where each pipeline stage lives

| Stage | Code location | Output location |
|-------|---------------|-----------------|
| **1. Prepare dataset** | `src/data_loader.py` (or a small script) | Raw images in `data/raw/` |
| **2. Preprocess dataset** | `src/data_loader.py` (the transform pipeline) | Pre‑processed tensors in `data/processed/` |
| **3. Set up DINO model** | `src/model.py` | No file output (model loaded in memory) |
| **4. Feature extraction** | `src/feature_extractor.py` | Feature vectors in `features/` (e.g., `features.npy`) |
| **5. Dimensionality reduction** | `src/dim_reducer.py` | Reduced embeddings in `embeddings/` |
| **6. Clustering** | `src/clustering.py` | Cluster labels in `clusters/` |
| **7. Evaluation & visualisation** | `src/utils.py` or a notebook in `notebooks/` | Plots in `plots/`; metrics can be logged in `results/summary.md` |
| **8. Final results** | Manual or script that aggregates previous artefacts | Summary report in `results/summary.md` and optionally a CSV in `results/` |

### Pre‑processing specifics
- Implement the transform pipeline (resize, center‑crop 224 × 224, augmentations, normalization) inside **`src/data_loader.py`**.
- The script should read each file from **`data/raw/`**, apply the transforms, and `torch.save` the tensor to **`data/processed/`** with the same base name but a `.pt` extension.
- `data/processed/` is treated as generated data; add it to `.gitignore` so it isn’t committed.

## Why this layout?
* **Separation of concerns** – raw data, generated artefacts, code, and results are isolated, making it easy to clean or re‑run any stage without affecting others.
* **Git‑friendly** – only source code and documentation are version‑controlled; large binary artefacts stay out of the repo.
* **Scalability** – each stage reads from a deterministic directory, so you can parallelise or cache intermediate results without rewriting code.
* **Reproducibility** – the `manifest.csv` (optional) records metadata such as original file names, timestamps, or ground‑truth labels, which are handy for evaluation.

---

**Next steps**
1. Add the `.gitignore` entries for the generated folders.
2. Scaffold the `src/` package with the files listed above.
3. Populate `data/raw/` with your images and start the preprocessing script.

Feel free to adjust any folder names to match your existing conventions; just keep the logical separation (raw → processed → features → embeddings → clusters → plots → results).
