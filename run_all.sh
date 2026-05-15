#!/bin/bash

# Pastikan script berhenti jika ada error
set -e

echo "=========================================================="
echo "      DINO ViT-S16 CLUSTERING PIPELINE RUNNER           "
echo "=========================================================="

# 1. Aktifkan Environment
echo "[1/4] Activating environment..."
source dino-env/bin/activate

# 2. Dimensionality Reduction
echo "[2/4] Running Dimensionality Reduction (UMAP 2D)..."
python3 src/dim_reducer.py \
  --features features/resized_256x256/features.npy \
  --output-dir embeddings/resized_256x256 \
  --method umap \
  --n-components 2

# 3. Clustering
echo "[3/4] Running K-Means Clustering (K=8)..."
python3 src/clustering.py \
  --embeddings embeddings/resized_256x256/embeddings_umap_2d.npy \
  --output-dir clusters/resized_256x256 \
  --method kmeans \
  --n-clusters 8 \
  --save-plots

# 4. Visualization
echo "[4/5] Generating Visualizations and Samples..."
python3 visualize_clusters.py \
  --embeddings embeddings/resized_256x256/embeddings_umap_2d.npy \
  --clusters clusters/resized_256x256/cluster_assignments_kmeans_8clusters.csv \
  --metadata features/resized_256x256/metadata.json \
  --output-dir plots/resized_256x256 \
  --raw-images data/raw/resized_datasets/resized_256x256 \
  --samples 15

# 5. Cluster Comparison Grid
echo "[5/5] Generating Cluster Comparison Grid..."
python3 visualize_comparison.py \
  --clusters clusters/resized_256x256/cluster_assignments_kmeans_8clusters.csv \
  --metadata features/resized_256x256/metadata.json \
  --raw-images data/raw/resized_datasets/resized_256x256 \
  --output-dir plots/resized_256x256 \
  --samples 6

echo "=========================================================="
echo "PIPELINE COMPLETE!"
echo "Check results in:"
echo " - Scatter Plot & Samples: plots/resized_256x256/"
echo " - Comparison Grid:       plots/resized_256x256/cluster_comparison_grid.png"
echo " - Summary Report:         results/summary.md"
echo "=========================================================="
