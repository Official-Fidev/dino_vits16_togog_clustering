# Project Workflow Diagrams: DINO ViT-S16 Clustering

This document provides visual flowcharts for each stage of the pipeline to help explain the internal processes of the system.

## 1. Overall Pipeline Flow
The high-level view of how data moves through the system.

```mermaid
graph TD
    A[Raw Images] --> B[Stage 1: Preprocessing]
    B --> C[Stage 2: Feature Extraction]
    C --> D[Stage 3: Dimensionality Reduction]
    D --> E[Stage 4: Unsupervised Clustering]
    E --> F[Stage 5: Visualization & Reporting]
```

---

## 2. Stage 1: Preprocessing Details
What happens inside `src/data_loader.py` and `preprocess.py`.

```mermaid
graph LR
    Input[Original Image] --> Resize[Resize to 256x256]
    Resize --> Crop[Center Crop to 224x224]
    Crop --> Tensor[Convert to PyTorch Tensor]
    Tensor --> Norm[Normalize - ImageNet Stats]
    Norm --> Save[Save as .pt File]
```

---

## 3. Stage 2: Feature Extraction Details
What happens inside `src/feature_extractor.py`.

```mermaid
graph TD
    PT[Load .pt Tensors] --> Model[Load DINO ViT-S16 Weights]
    Model --> Device[Detect GPU/CPU Device]
    Device --> Forward[Forward Pass - No Gradient]
    Forward --> Gap[Global Average Pooling]
    Gap --> Vector[768-Dim Feature Vector]
    Vector --> SaveNPY[Save features.npy]
```

---

## 4. Stage 3: Dimensionality Reduction (UMAP)
What happens inside `src/dim_reducer.py`.

```mermaid
graph LR
    NPY[features.npy - 768D] --> UMAP[Fit UMAP Reducer]
    UMAP --> Params[n_neighbors=15, min_dist=0.1]
    Params --> Transform[Project to 2D Space]
    Transform --> Embed[embeddings_2d.npy]
```

---

## 5. Stage 4: Unsupervised Clustering (K-Means)
What happens inside `src/clustering.py`.

```mermaid
graph TD
    Embed[embeddings_2d.npy] --> KSelect[Set K=8 Clusters]
    KSelect --> KMeans[Fit K-Means Algorithm]
    KMeans --> Labels[Generate Cluster Labels]
    Labels --> Eval[Calculate Silhouette Score]
    Eval --> CSV[Save cluster_assignments.csv]
```

---

## 6. Stage 5: Visualization & Comparison
What happens inside the visualization scripts.

```mermaid
graph TD
    Labels[Labels] --> Scatter[Generate Scatter Plot]
    Labels --> Sampling[Sample 15 Images per Cluster]
    Sampling --> Montage[Create Grid Comparison Grid]
    Montage --> Final[Final Results & Plots]
```
