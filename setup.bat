@echo off
setlocal

cd /d "%~dp0"

echo ==========================================================
echo       SETUP: DINO ViT-S/16 CLUSTERING (Windows)
echo ==========================================================

REM ---------- 1. Cek Python ----------
echo [1/5] Memeriksa Python...
where py >nul 2>nul
if %errorlevel%==0 (
    set "PYLAUNCH=py -3"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Error: Python tidak ditemukan di PATH.
        echo Instal Python 3.12 dari python.org dan centang "Add python.exe to PATH".
        exit /b 1
    )
    set "PYLAUNCH=python"
)
%PYLAUNCH% --version

REM ---------- 2. Deteksi GPU ----------
echo [2/5] Mendeteksi hardware (GPU/CPU)...
set "HAS_CUDA=0"
where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
    set "HAS_CUDA=1"
    echo Driver NVIDIA ditemukan:
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    echo Build CUDA 12.1 akan dipakai. Build ini butuh driver versi 525 atau lebih baru.
) else (
    echo Driver NVIDIA tidak ditemukan. Menggunakan mode CPU.
)

REM ---------- 3. Virtual environment ----------
echo [3/5] Menyiapkan virtual environment (dino-env)...
if exist "dino-env\Scripts\activate.bat" (
    echo Virtual environment sudah ada. Melewati pembuatan.
) else (
    %PYLAUNCH% -m venv dino-env
    if errorlevel 1 (
        echo Error: gagal membuat virtual environment.
        exit /b 1
    )
    echo Virtual environment berhasil dibuat.
)

call "dino-env\Scripts\activate.bat"
if errorlevel 1 (
    echo Error: gagal mengaktifkan virtual environment.
    exit /b 1
)

python -m pip install --upgrade pip --quiet

REM ---------- 4. PyTorch ----------
echo [4/5] Menginstal PyTorch...
if "%HAS_CUDA%"=="1" (
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
) else (
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
)
if errorlevel 1 (
    echo Error: instalasi PyTorch gagal.
    exit /b 1
)

REM ---------- 5. Dependensi lain ----------
echo [5/5] Menginstal dependensi lain dari requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Error: instalasi dependensi gagal.
    echo Penyebab paling umum: paket hdbscan perlu dikompilasi karena tidak ada
    echo wheel siap pakai untuk versi Python ini. Solusinya pilih salah satu:
    echo   a^) pakai Python 3.12 ^(wheel hdbscan tersedia^), atau
    echo   b^) instal "Microsoft C++ Build Tools" lalu jalankan ulang setup.bat.
    exit /b 1
)

echo.
echo ----------------------------------------------------------
echo Verifikasi instalasi...
python -c "import torch, timm, umap, hdbscan, optuna; print('torch', torch.__version__, '| CUDA tersedia:', torch.cuda.is_available())"
if errorlevel 1 (
    echo Error: verifikasi import gagal.
    exit /b 1
)

echo ----------------------------------------------------------
echo INSTALASI SELESAI
echo ----------------------------------------------------------
echo Langkah berikutnya:
echo   1. Pastikan citra sudah ada di: data\raw\resized_datasets\resized_256x256
echo   2. Jalankan seluruh pipeline dengan: run_all.bat
echo.
echo Untuk bekerja manual, aktifkan environment dulu:
echo   dino-env\Scripts\activate
echo ----------------------------------------------------------
endlocal
