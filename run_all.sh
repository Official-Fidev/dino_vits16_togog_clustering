#!/bin/bash

# Pastikan script berhenti jika ada error
set -e

echo "=========================================================="
echo "      DINO ViT-S16 CLUSTERING PIPELINE RUNNER           "
echo "=========================================================="

# 1. Aktifkan Environment
echo "[1/7] Activating environment..."
source dino-env/bin/activate

# 2. Preprocessing
echo "[2/7] Running Preprocessing..."
python3 preprocess.py

# 3. Feature Extraction
echo "[3/7] Running Feature Extraction..."
python3 extract_features.py

# 4. Dimensionality Reduction
echo "[4/7] Running Dimensionality Reduction (UMAP 2D)..."
python3 src/dim_reducer.py \
  --features outputs/features/resized_256x256/features.npy \
  --output-dir outputs/embeddings/resized_256x256 \
  --method umap \
  --n-components 2

# 5. Clustering (HDBSCAN)
echo "[5/7] Running HDBSCAN Clustering (Automatic K)..."
python3 src/clustering.py \
  --embeddings outputs/embeddings/resized_256x256/embeddings_umap_2d.npy \
  --output-dir outputs/clusters/resized_256x256 \
  --method hdbscan \
  --min-cluster-size 5 \
  --save-plots

# 6. Visualization
echo "[6/7] Generating Visualizations and Samples..."
python3 visualize_clusters.py \
  --embeddings outputs/embeddings/resized_256x256/embeddings_umap_2d.npy \
  --clusters outputs/clusters/resized_256x256/cluster_assignments_hdbscan.csv \
  --metadata outputs/features/resized_256x256/metadata.json \
  --output-dir outputs/plots/resized_256x256 \
  --raw-images data/raw/resized_datasets/resized_256x256 \
  --samples 15

# 7. Cluster Comparison Grid
echo "[7/7] Generating Cluster Comparison Grid..."
python3 visualize_comparison.py \
  --clusters outputs/clusters/resized_256x256/cluster_assignments_hdbscan.csv \
  --metadata outputs/features/resized_256x256/metadata.json \
  --raw-images data/raw/resized_datasets/resized_256x256 \
  --output-dir outputs/plots/resized_256x256 \
  --samples 6

echo "=========================================================="
echo "PIPELINE COMPLETE!"
echo "Check results in:"
echo " - Scatter Plot & Samples: outputs/plots/resized_256x256/"
echo " - Comparison Grid:       outputs/plots/resized_256x256/cluster_comparison_grid.png"
echo " - Summary Report:         outputs/results/summary.md"
echo "=========================================================="
