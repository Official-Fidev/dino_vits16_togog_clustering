@echo off
setlocal

cd /d "%~dp0"

echo ==========================================================
echo       SETUP: DINO ViT-S/16 CLUSTERING (Windows)
echo ==========================================================

REM ---------- 1. Cek Python ----------
REM Wajib Python 3.12. numpy 2.5.4 dan scipy 1.18.1 butuh >= 3.12,
REM sedangkan torch 2.5.1 tidak punya wheel untuk 3.13 ke atas.
REM "python" dicoba lebih dulu supaya conda env yang sedang aktif terpakai.
echo [1/5] Memeriksa Python...
set "PYLAUNCH="
call :coba python
call :coba py -3.12
call :coba python3.12

REM Belum ada 3.12 di PATH. Kalau conda tersedia, siapkan sendiri.
if not defined PYLAUNCH call :siapkan_conda

if not defined PYLAUNCH (
    echo.
    echo Error: Python 3.12 tidak ditemukan.
    echo.
    echo Python yang terdeteksi sekarang:
    python --version 2>nul || echo   ^(python tidak ada di PATH^)
    py -0 2>nul
    echo.
    echo Proyek ini butuh Python 3.12 tepat: numpy 2.5.4 dan scipy 1.18.1
    echo mensyaratkan 3.12 ke atas, sedangkan torch 2.5.1 tidak tersedia
    echo untuk 3.13 ke atas. Siapkan 3.12 lalu jalankan setup.bat lagi.
    echo.
    echo Conda tidak terdeteksi, atau pembuatan env py312 gagal.
    echo Pasang salah satu lalu jalankan setup.bat lagi:
    echo   Miniconda   : https://docs.conda.io/en/latest/miniconda.html
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

REM dino-env sisa percobaan lama bisa memakai Python yang salah.
REM Periksa versinya, dan bangun ulang bila tidak cocok.
if exist "dino-env\Scripts\python.exe" (
    dino-env\Scripts\python.exe -c "import sys; sys.exit(0 if sys.version_info[:2] == (3,12) else 1)" >nul 2>nul
    if errorlevel 1 (
        echo dino-env lama tidak memakai Python 3.12. Membangun ulang...
        rmdir /s /q dino-env
    )
)

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
python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3,12) else 1)" >nul 2>nul
if errorlevel 1 (
    echo.
    echo Error: dino-env tidak memakai Python 3.12:
    python --version
    echo.
    echo Ini tidak seharusnya terjadi. Hapus foldernya lalu ulangi:
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
REM  Subrutin: pakai kandidat interpreter bila versinya tepat 3.12
REM  %* = perintah yang diuji, mis. "python" atau "py -3.12"
REM =========================================================
:coba
if defined PYLAUNCH exit /b 0
%* -c "import sys; sys.exit(0 if sys.version_info[:2] == (3,12) else 1)" >nul 2>nul
if not errorlevel 1 set "PYLAUNCH=%*"
exit /b 0

REM =========================================================
REM  Subrutin: buat conda env py312 lalu pakai python.exe-nya
REM  langsung, tanpa perlu "conda activate" dari pengguna.
REM =========================================================
:siapkan_conda
where conda >nul 2>nul
if errorlevel 1 exit /b 0

REM Buat env bila belum terdaftar. Aman diulang.
conda env list | findstr /r /c:"[\\/]py312 " /c:"[\\/]py312$" >nul 2>nul
if errorlevel 1 (
    echo Python 3.12 belum ada. Membuat conda env "py312", mohon tunggu...
    conda create -y -n py312 python=3.12
    if errorlevel 1 (
        echo Peringatan: conda create gagal.
        exit /b 0
    )
)

REM Tanyakan jalur interpreternya langsung ke conda, jangan menebak,
REM karena folder envs bisa saja tidak berada di bawah conda base.
set "PY312="
for /f "usebackq delims=" %%P in (`conda run -n py312 python -c "import sys; print(sys.executable)" 2^>nul`) do set "PY312=%%P"

REM Cadangan: tebak lewat conda base kalau conda run tidak memberi apa-apa.
if not defined PY312 (
    for /f "usebackq delims=" %%B in (`conda info --base 2^>nul`) do set "CONDA_BASE=%%B"
    call :pakai_base
)

if not defined PY312 (
    echo Peringatan: env py312 ada tetapi jalur python.exe tidak terbaca.
    exit /b 0
)
if not exist "%PY312%" (
    echo Peringatan: tidak menemukan %PY312%
    exit /b 0
)

echo Memakai Python 3.12 dari conda: %PY312%
call :coba "%PY312%"
exit /b 0

REM Dipanggil terpisah supaya %CONDA_BASE% dibaca setelah for /f selesai.
:pakai_base
if not defined CONDA_BASE exit /b 0
if exist "%CONDA_BASE%\envs\py312\python.exe" set "PY312=%CONDA_BASE%\envs\py312\python.exe"
exit /b 0
