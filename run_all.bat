@echo off
echo ==========================================================
echo       DINO ViT-S16 CLUSTERING PIPELINE RUNNER           
echo ==========================================================

echo [1/9] Running Preprocessing...
python preprocess.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/9] Running Feature Extraction...
python extract_features.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3/9] Running Dimensionality Reduction (UMAP 10D ^& 2D)...
python src\dim_reducer.py --features outputs\features\resized_256x256\features.npy --output-dir outputs\embeddings\resized_256x256 --method umap --n-components 10 --n-neighbors 30 --min-dist 0.0 --metric cosine
if %errorlevel% neq 0 exit /b %errorlevel%
python src\dim_reducer.py --features outputs\features\resized_256x256\features.npy --output-dir outputs\embeddings\resized_256x256 --method umap --n-components 2 --n-neighbors 30 --min-dist 0.0 --metric cosine
if %errorlevel% neq 0 exit /b %errorlevel%

echo [4/9] Running Optuna Auto-tuning for HDBSCAN...
python src\auto_tune.py --embeddings outputs\embeddings\resized_256x256\embeddings_umap_10d.npy --output outputs\results\best_params.json --n-trials 30
if %errorlevel% neq 0 exit /b %errorlevel%

echo [5/9] Running HDBSCAN Clustering with best parameters...
python src\clustering.py --embeddings outputs\embeddings\resized_256x256\embeddings_umap_10d.npy --embeddings-2d outputs\embeddings\resized_256x256\embeddings_umap_2d.npy --output-dir outputs\clusters\resized_256x256 --method hdbscan --params-file outputs\results\best_params.json --save-plots
if %errorlevel% neq 0 exit /b %errorlevel%

echo [6/9] Generating Visualizations and Samples...
python visualize_clusters.py --embeddings outputs\embeddings\resized_256x256\embeddings_umap_2d.npy --clusters outputs\clusters\resized_256x256\cluster_assignments_hdbscan.csv --metadata outputs\features\resized_256x256\metadata.json --output-dir outputs\plots\resized_256x256 --raw-images data\raw\resized_datasets\resized_256x256 --samples 15
if %errorlevel% neq 0 exit /b %errorlevel%

echo [7/9] Generating Cluster Comparison Grid...
python visualize_comparison.py --clusters outputs\clusters\resized_256x256\cluster_assignments_hdbscan.csv --metadata outputs\features\resized_256x256\metadata.json --raw-images data\raw\resized_datasets\resized_256x256 --output-dir outputs\plots\resized_256x256 --samples 6
if %errorlevel% neq 0 exit /b %errorlevel%

echo [8/9] Cleaning up temp files...
echo (Optional cleanup steps here)

:: 9. Save to History
echo [9/9] Saving outputs to history folder...
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set TIMESTAMP=%datetime:~0,8%_%datetime:~8,6%
set HISTORY_DIR=outputs\history\run_%TIMESTAMP%
mkdir "%HISTORY_DIR%"

xcopy /E /I /Y outputs\clusters "%HISTORY_DIR%\clusters"
xcopy /E /I /Y outputs\plots "%HISTORY_DIR%\plots"
xcopy /E /I /Y outputs\results "%HISTORY_DIR%\results"

echo ==========================================================
echo PIPELINE COMPLETE!
echo Check results in:
echo  - Scatter Plot ^& Samples: outputs\plots\resized_256x256\
echo  - Comparison Grid:       outputs\plots\resized_256x256\cluster_comparison_grid.png
echo  - HISTORY SAVED TO:       %HISTORY_DIR%
echo ==========================================================
pause
