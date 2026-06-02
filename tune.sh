#!/bin/bash

# Pastikan script berhenti jika ada error
set -e

echo "=========================================================="
echo "          HDBSCAN GRID SEARCH TUNING (10D)                "
echo "=========================================================="

# 1. Aktifkan Environment
echo "Activating environment..."
source dino-env/bin/activate

# 2. Run Tuning Script
echo "Running tuning script (this may take a minute)..."
python3 src/tune_hdbscan.py --embeddings outputs/embeddings/resized_256x256/embeddings_umap_10d.npy

echo "=========================================================="
echo "TUNING COMPLETE!"
echo "Check results in: outputs/results/hdbscan_tuning.json"
echo "=========================================================="
