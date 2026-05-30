@echo off
echo ==========================================================
echo       DINO ViT-S16 CLUSTERING PIPELINE RUNNER           
echo ==========================================================

echo [1/6] Running Preprocessing...
python preprocess.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/6] Running Feature Extraction...
python extract_features.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3/6] Running Dimensionality Reduction (UMAP 2D)...
python src\dim_reducer.py --features outputs\features\resized_256x256\features.npy --output-dir outputs\embeddings\resized_256x256 --method umap --n-components 2
if %errorlevel% neq 0 exit /b %errorlevel%

echo [4/6] Running HDBSCAN Clustering (Automatic K)...
python src\clustering.py --embeddings outputs\embeddings\resized_256x256\embeddings_umap_2d.npy --output-dir outputs\clusters\resized_256x256 --method hdbscan --min-cluster-size 5 --save-plots
if %errorlevel% neq 0 exit /b %errorlevel%

echo [5/6] Generating Visualizations and Samples...
python visualize_clusters.py --embeddings outputs\embeddings\resized_256x256\embeddings_umap_2d.npy --clusters outputs\clusters\resized_256x256\cluster_assignments_hdbscan.csv --metadata outputs\features\resized_256x256\metadata.json --output-dir outputs\plots\resized_256x256 --raw-images data\raw\resized_datasets\resized_256x256 --samples 15
if %errorlevel% neq 0 exit /b %errorlevel%

echo [6/6] Generating Cluster Comparison Grid...
python visualize_comparison.py --clusters outputs\clusters\resized_256x256\cluster_assignments_hdbscan.csv --metadata outputs\features\resized_256x256\metadata.json --raw-images data\raw\resized_datasets\resized_256x256 --output-dir outputs\plots\resized_256x256 --samples 6
if %errorlevel% neq 0 exit /b %errorlevel%

echo ==========================================================
echo PIPELINE COMPLETE!
echo Check results in:
echo  - Scatter Plot ^& Samples: outputs\plots\resized_256x256\
echo  - Comparison Grid:       outputs\plots\resized_256x256\cluster_comparison_grid.png
echo  - Summary Report:         outputs\results\summary.md
echo ==========================================================
pause
