# Project Workflow Diagrams: DINO ViT-S16 HDBSCAN Clustering (Auto-Tuned)

This document provides flowcharts reflecting the current **Density-Based Automatic Clustering** feature using UMAP 10D, Optuna Auto-tuning, and HDBSCAN, based on the latest pipeline execution architecture.

## 1. Overall Pipeline Flow
The high-level view showing all stages from raw images to generating visual comparison grids and saving history.

```mermaid
graph TD
    A[Raw Images] --> B[Stage 1: Preprocessing]
    B --> C[Stage 2: Feature Extraction]
    C --> D[Stage 3: Dimensional Reduction 10D & 2D]
    D --> E[Stage 4: Optuna Auto-tuning for HDBSCAN]
    E --> F[Stage 5: HDBSCAN Density Clustering]
    F --> G[Stage 6: Visualizations & Comparison Grids]
    G --> H[Stage 7: Save to History & Cleanup]
```

---

## 2. Stage 3, 4, & 5: Dimensionality Reduction, Tuning & Clustering Detail
Detailed logic of how features are reduced, parameters are tuned, and clustering is applied.

```mermaid
graph TD
    Feat[features.npy] --> DR10[UMAP 10D: Clustering Space]
    Feat --> DR2[UMAP 2D: Visualization Space]
    
    DR10 --> AutoTune[Optuna Auto-Tuning]
    
    subgraph "Optuna Trials"
    AutoTune -.-> Param1[min_cluster_size]
    AutoTune -.-> Param2[min_samples]
    AutoTune -.-> Eval[Evaluate DBCV & Silhouette]
    end
    
    Eval --> BestParams[best_params.json]
    
    BestParams --> HDB[Initialize HDBSCAN with Best Params]
    DR10 --> HDB[Fit HDBSCAN on 10D]
    
    HDB --> Split{Data Density Check}
    
    Split -->|Dense Region| Core[Assign to Cluster ID 0, 1, 2...]
    Split -->|Sparse/Isolated| Outlier[Assign as Outlier Label: -1]
    
    Core --> OutDir[Save cluster_assignments_hdbscan.csv]
    Outlier --> OutDir
```

---

## 3. Stage 6 & 7: Visualization, Reporting & History
How the outputs are visualized, dealing with UMAP 2D, generating comparison grids, and archiving.

```mermaid
graph TD
    Labels[cluster_assignments_hdbscan.csv] --> Filter{Filter Outliers}
    U2D[embeddings_umap_2d.npy] --> Scatter[Generate Scatter Plot]
    Filter --> Scatter
    
    Filter -->|Valid Clusters| ScatterPlot[Plot Colored Points]
    Filter -->|Outliers| GreyPoints[Plot as Grey Noise Background]
    
    Filter -->|Valid Clusters| Sampling[Sample Images per Cluster]
    Sampling --> Montage[Create Comparison Grid]
    
    ScatterPlot --> PlotOut[Save cluster_scatter.png]
    Montage --> GridOut[Save cluster_comparison_grid.png]
    
    PlotOut --> Archive[Save to History Folder: run_TIMESTAMP/]
    GridOut --> Archive
    Archive --> Final[Final Results & Plots Ready]
```
