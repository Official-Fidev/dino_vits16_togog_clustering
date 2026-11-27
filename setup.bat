@echo off
setlocal

cd /d "%~dp0"

echo ==========================================================
echo       SETUP: DINO ViT-S/16 CLUSTERING (Windows)
echo ==========================================================

REM ---------- 1. Cek Python ----------
REM Torch 2.5.1 dan hdbscan 0.8.44 hanya punya wheel sampai Python 3.12,
REM jadi interpreter dipilih berdasarkan versinya, bukan yang paling baru.
REM "python" dicoba lebih dulu supaya conda env yang sedang aktif terpakai.
echo [1/5] Memeriksa Python...
set "PYLAUNCH="
call :coba python
call :coba py -3.12
call :coba py -3.11
call :coba py -3.10
call :coba py -3

if not defined PYLAUNCH (
    echo.
    echo Error: tidak ditemukan Python 3.10, 3.11, atau 3.12.
    echo.
    echo Python yang terdeteksi sekarang:
    python --version 2>nul || echo   ^(python tidak ada di PATH^)
    py -0 2>nul
    echo.
    echo Proyek ini memakai torch 2.5.1 dan hdbscan 0.8.44, yang belum punya
    echo wheel untuk Python 3.13 ke atas. Siapkan Python 3.12 lalu ulangi.
    echo.
    echo   Pakai conda : conda create -y -n py312 python=3.12 ^&^& conda activate py312
    echo   Tanpa conda : instal dari https://www.python.org/downloads/
    exit /b 1
)

echo Memakai interpreter: %PYLAUNCH%
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

REM Venv yang sudah ada bisa saja dibuat dengan Python yang tidak cocok
REM (mis. 3.14 dari percobaan sebelumnya). Periksa dan hentikan lebih awal.
python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info < (3,13) else 1)" >nul 2>nul
if errorlevel 1 (
    echo.
    echo Error: dino-env memakai Python yang tidak didukung:
    python --version
    echo.
    echo Hapus foldernya lalu jalankan setup.bat lagi:
    echo   rmdir /s /q dino-env
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
exit /b 0

REM =========================================================
REM  Subrutin: pakai kandidat interpreter bila versinya cocok
REM  %* = perintah yang diuji, mis. "python" atau "py -3.12"
REM =========================================================
:coba
if defined PYLAUNCH exit /b 0
%* -c "import sys; sys.exit(0 if (3,10) <= sys.version_info < (3,13) else 1)" >nul 2>nul
if not errorlevel 1 set "PYLAUNCH=%*"
exit /b 0
