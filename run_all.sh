#!/bin/bash

# Pastikan script berhenti jika ada error
set -e

NUM_RUNS=10

echo "=========================================================="
echo "      DINO ViT-S16 CLUSTERING PIPELINE RUNNER           "
echo "      (Running $NUM_RUNS Iterations for Comparison)       "
echo "=========================================================="

# 1. Aktifkan Environment
echo "[1/9] Activating environment..."
source dino-env/bin/activate

# 2. Preprocessing (Only needs to run once)
echo "[2/9] Running Preprocessing..."
python3 preprocess.py

# 3. Feature Extraction (Only needs to run once)
echo "[3/9] Running Feature Extraction..."
python3 extract_features.py

for i in $(seq 1 $NUM_RUNS); do
    echo ""
    echo "=========================================================="
    echo "      STARTING RUN ITERATION $i OF $NUM_RUNS"
    echo "=========================================================="


# 4. Dimensionality Reduction (UMAP 10D for Clustering & 2D for Viz)
echo "[4/9] Running Dimensionality Reduction (UMAP 10D & 2D)..."
python3 src/dim_reducer.py \
  --features outputs/features/resized_256x256/features.npy \
  --output-dir outputs/embeddings/resized_256x256 \
  --method umap \
  --n-components 10 \
  --n-neighbors 30 \
  --min-dist 0.0 \
  --metric cosine

python3 src/dim_reducer.py \
  --features outputs/features/resized_256x256/features.npy \
  --output-dir outputs/embeddings/resized_256x256 \
  --method umap \
  --n-components 2 \
  --n-neighbors 30 \
  --min-dist 0.0 \
  --metric cosine

# 5. Auto-tuning HDBSCAN (Optuna)
echo "[5/9] Running Optuna Auto-tuning for HDBSCAN..."
python3 src/auto_tune.py \
  --embeddings outputs/embeddings/resized_256x256/embeddings_umap_10d.npy \
  --output outputs/results/best_params.json \
  --n-trials 30

# 6. Clustering (HDBSCAN)
echo "[6/9] Running HDBSCAN Clustering with best parameters..."
python3 src/clustering.py \
  --embeddings outputs/embeddings/resized_256x256/embeddings_umap_10d.npy \
  --embeddings-2d outputs/embeddings/resized_256x256/embeddings_umap_2d.npy \
  --output-dir outputs/clusters/resized_256x256 \
  --method hdbscan \
  --params-file outputs/results/best_params.json \
  --save-plots

# 7. Visualization
echo "[7/9] Generating Visualizations and Samples..."
python3 visualize_clusters.py \
  --embeddings outputs/embeddings/resized_256x256/embeddings_umap_2d.npy \
  --clusters outputs/clusters/resized_256x256/cluster_assignments_hdbscan.csv \
  --metadata outputs/features/resized_256x256/metadata.json \
  --output-dir outputs/plots/resized_256x256 \
  --raw-images data/raw/resized_datasets/resized_256x256 \
  --samples 15

# 8. Cluster Comparison Grid
echo "[8/9] Generating Cluster Comparison Grid..."
python3 visualize_comparison.py \
  --clusters outputs/clusters/resized_256x256/cluster_assignments_hdbscan.csv \
  --metadata outputs/features/resized_256x256/metadata.json \
  --raw-images data/raw/resized_datasets/resized_256x256 \
  --output-dir outputs/plots/resized_256x256 \
  --samples 6

# 9. Save to History
    echo "[9/9] Saving outputs to history folder..."
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    HISTORY_DIR="outputs/history/run_iter${i}_${TIMESTAMP}"
    mkdir -p "${HISTORY_DIR}"

    # Salin file-file hasil ke folder history untuk perbandingan
    cp -r outputs/clusters "${HISTORY_DIR}/"
    cp -r outputs/plots "${HISTORY_DIR}/"
    cp -r outputs/results "${HISTORY_DIR}/"

    echo "Iteration $i saved to: ${HISTORY_DIR}"
    echo "----------------------------------------------------------"
done

echo "=========================================================="
echo "ALL $NUM_RUNS PIPELINE ITERATIONS COMPLETE!"
echo "Check your history folders in outputs/history/ for comparisons."
echo "=========================================================="
