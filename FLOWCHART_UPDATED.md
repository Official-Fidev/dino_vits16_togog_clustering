# Project Workflow Diagrams: DINO ViT-S16 HDBSCAN Clustering

This document provides flowcharts reflecting the **Density-Based Automatic Clustering** feature using HDBSCAN.

## 1. Overall Pipeline Flow (HDBSCAN)
The high-level view showing the shift to density-based clustering.

```mermaid
graph TD
    A[Raw Images] --> B[Stage 1: Preprocessing]
    B --> C[Stage 2: Feature Extraction]
    C --> D[Stage 3: Dimensionality Reduction]
    D --> E[Stage 4: HDBSCAN Density Clustering]
    E --> F[Stage 5: Visualization & Reporting]
```

---

## 2. Stage 4: HDBSCAN Clustering Detail
Detailed logic of how HDBSCAN processes the data without requiring a predefined number of clusters.

```mermaid
graph TD
    Embed[embeddings_2d.npy] --> HDB[Initialize HDBSCAN]
    
    subgraph "Parameters"
    HDB -.-> P1[min_cluster_size: e.g., 5]
    HDB -.-> P2[cluster_selection_method: 'eom']
    end
    
    HDB --> Fit[Fit & Predict]
    Fit --> Split{Data Density Check}
    
    Split -->|Dense Region| Core[Assign to Cluster ID 0, 1, 2...]
    Split -->|Sparse/Isolated| Outlier[Assign as Outlier Label: -1]
    
    Core --> Metrics[Calculate Evaluation Metrics]
    Outlier --> Metrics
    
    subgraph "Metrics Calculation"
    Metrics --> DBCV[DBCV Score: Density Validity]
    Metrics --> Sil[Silhouette Score: On non-outliers only]
    end
    
    DBCV --> Output[Generate Outputs]
    Sil --> Output
    
    Output --> Plot[Save hdbscan_result.png]
    Output --> CSV[Save cluster_assignments.csv]
```

---

## 3. Visualization & Documentation (Updated)
How the outputs are visualized, specifically handling noise/outliers.

```mermaid
graph TD
    Labels[Labels with -1 Outliers] --> Filter{Filter Outliers}
    
    Filter -->|Valid Clusters| Scatter[Generate Scatter Plot]
    Filter -->|Outliers| GreyPoints[Plot as Grey Noise Background]
    
    Filter -->|Valid Clusters| Sampling[Sample Images per Cluster]
    Sampling --> Montage[Create Comparison Grid]
    
    Plot[hdbscan_result.png] --> Report[Update results/summary.md]
    Montage --> Final[Final Results & Plots Ready]
```
