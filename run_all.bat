@echo off
setlocal

cd /d "%~dp0"

set NUM_RUNS=10

set "DATA_RAW=data\raw\resized_datasets\resized_256x256"
set "FEATURES=outputs\features\resized_256x256"
set "EMBEDDINGS=outputs\embeddings\resized_256x256"
set "CLUSTERS=outputs\clusters\resized_256x256"
set "PLOTS=outputs\plots\resized_256x256"
set "RESULTS=outputs\results"

echo ==========================================================
echo       DINO ViT-S/16 CLUSTERING PIPELINE RUNNER
echo       (%NUM_RUNS% iterasi untuk perbandingan)
echo ==========================================================

REM ---------- 1. Aktifkan environment ----------
echo [1/9] Mengaktifkan environment...
if not exist "dino-env\Scripts\activate.bat" (
    echo Error: virtual environment tidak ditemukan.
    echo Jalankan setup.bat terlebih dahulu.
    exit /b 1
)
call "dino-env\Scripts\activate.bat"
if errorlevel 1 (
    echo Error: gagal mengaktifkan environment.
    exit /b 1
)

REM ---------- 2. Preprocessing (cukup sekali) ----------
echo [2/9] Menjalankan preprocessing...
python preprocess.py
if errorlevel 1 exit /b 1

REM ---------- 3. Ekstraksi fitur (cukup sekali) ----------
echo [3/9] Menjalankan ekstraksi fitur DINO ViT-S/16...
python extract_features.py
if errorlevel 1 exit /b 1

REM ---------- Iterasi 4..9 ----------
for /L %%i in (1,1,%NUM_RUNS%) do (
    call :run_iteration %%i
    if errorlevel 1 (
        echo Error: iterasi %%i gagal. Pipeline dihentikan.
        exit /b 1
    )
)

echo ==========================================================
echo SELURUH %NUM_RUNS% ITERASI SELESAI
echo Bandingkan hasilnya di: outputs\history\
echo ==========================================================
endlocal
exit /b 0


REM =========================================================
REM  Subrutin: satu iterasi lengkap (langkah 4 sampai 9)
REM  %1 = nomor iterasi
REM =========================================================
:run_iteration
set "ITER=%~1"

echo.
echo ==========================================================
echo       MULAI ITERASI %ITER% DARI %NUM_RUNS%
echo ==========================================================

REM ---------- 4. Reduksi dimensi (UMAP 10D untuk klasterisasi, 2D untuk visualisasi) ----------
echo [4/9] Menjalankan reduksi dimensi (UMAP 10D ^& 2D)...
python src\dim_reducer.py --features "%FEATURES%\features.npy" --output-dir "%EMBEDDINGS%" --method umap --n-components 10 --n-neighbors 30 --min-dist 0.0 --metric cosine
if errorlevel 1 exit /b 1

python src\dim_reducer.py --features "%FEATURES%\features.npy" --output-dir "%EMBEDDINGS%" --method umap --n-components 2 --n-neighbors 30 --min-dist 0.0 --metric cosine
if errorlevel 1 exit /b 1

REM ---------- 5. Auto-tuning HDBSCAN (Optuna) ----------
echo [5/9] Menjalankan auto-tuning Optuna untuk HDBSCAN...
python src\auto_tune.py --embeddings "%EMBEDDINGS%\embeddings_umap_10d.npy" --output "%RESULTS%\best_params.json" --n-trials 30
if errorlevel 1 exit /b 1

REM ---------- 6. Klasterisasi HDBSCAN ----------
echo [6/9] Menjalankan klasterisasi HDBSCAN dengan parameter terbaik...
python src\clustering.py --embeddings "%EMBEDDINGS%\embeddings_umap_10d.npy" --embeddings-2d "%EMBEDDINGS%\embeddings_umap_2d.npy" --output-dir "%CLUSTERS%" --method hdbscan --params-file "%RESULTS%\best_params.json" --save-plots
if errorlevel 1 exit /b 1

REM ---------- 7. Visualisasi scatter plot dan sampel ----------
echo [7/9] Membuat visualisasi dan sampel klaster...
python visualize_clusters.py --embeddings "%EMBEDDINGS%\embeddings_umap_2d.npy" --clusters "%CLUSTERS%\cluster_assignments_hdbscan.csv" --metadata "%FEATURES%\metadata.json" --output-dir "%PLOTS%" --raw-images "%DATA_RAW%" --samples 15
if errorlevel 1 exit /b 1

REM ---------- 8. Grid perbandingan klaster ----------
echo [8/9] Membuat grid perbandingan klaster...
python visualize_comparison.py --clusters "%CLUSTERS%\cluster_assignments_hdbscan.csv" --metadata "%FEATURES%\metadata.json" --raw-images "%DATA_RAW%" --output-dir "%PLOTS%" --samples 6
if errorlevel 1 exit /b 1

REM ---------- 9. Simpan ke history ----------
echo [9/9] Menyimpan keluaran ke folder history...
set "TIMESTAMP="
for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"`) do set "TIMESTAMP=%%T"
if not defined TIMESTAMP (
    echo Peringatan: gagal membaca waktu dari PowerShell, memakai penanda cadangan.
    set "TIMESTAMP=run%ITER%"
)

set "HISTORY_DIR=outputs\history\run_iter%ITER%_%TIMESTAMP%"
if not exist "%HISTORY_DIR%" mkdir "%HISTORY_DIR%"

xcopy /E /I /Y /Q "outputs\clusters" "%HISTORY_DIR%\clusters" >nul
xcopy /E /I /Y /Q "outputs\plots"    "%HISTORY_DIR%\plots"    >nul
xcopy /E /I /Y /Q "outputs\results"  "%HISTORY_DIR%\results"  >nul

echo Iterasi %ITER% disimpan di: %HISTORY_DIR%
echo ----------------------------------------------------------
exit /b 0
