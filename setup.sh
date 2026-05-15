#!/bin/bash

# ==========================================================
#      SETUP SCRIPT: DINO ViT-S16 CLUSTERING
# ==========================================================
# Deskripsi: Skrip otomatis untuk instalasi environment 
#            dan dependensi project.
# ==========================================================

set -e

echo "----------------------------------------------------------"
echo " Menyiapkan Lingkungan Kerja (Setup Project)..."
echo "----------------------------------------------------------"

# 1. Cek Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 tidak ditemukan. Silakan instal Python3 terlebih dahulu."
    exit 1
fi

# 2. Deteksi CUDA
echo "[1/4] Mendeteksi hardware (GPU/CPU)..."
CUDA_VERSION=""
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | awk '{print $1}')
    CUDA_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d'.' -f1)
    echo "Found NVIDIA Driver version: $CUDA_VERSION"
else
    echo "NVIDIA Driver tidak ditemukan. Menggunakan mode CPU."
fi

# 3. Buat Virtual Environment
echo "[2/4] Membuat Virtual Environment (dino-env)..."
if [ -d "dino-env" ]; then
    echo "Virtual environment 'dino-env' sudah ada. Melewati langkah ini."
else
    python3 -m venv dino-env
    echo "Virtual environment berhasil dibuat."
fi

# Aktifkan Environment
source dino-env/bin/activate

# 4. Upgrade Pip
pip install --upgrade pip --quiet

# 5. Instalasi Library
echo "[3/4] Menginstal library..."

# Logika instalasi Torch berdasarkan CUDA
if [ ! -z "$CUDA_VERSION" ]; then
    # Jika CUDA ditemukan, tanya user atau deteksi versi major
    if [ "$CUDA_MAJOR" -ge 525 ]; then
        echo "Menginstal PyTorch untuk CUDA 12.1..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    elif [ "$CUDA_MAJOR" -ge 450 ]; then
        echo "Menginstal PyTorch untuk CUDA 11.8..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        echo "Versi Driver lama. Menginstal PyTorch versi default GPU..."
        pip install torch torchvision
    fi
else
    echo "Menginstal PyTorch versi CPU..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Instal dependensi umum
echo "Menginstal dependensi tambahan (Scikit-learn, UMAP, Pandas, dll)..."
pip install numpy pandas scikit-learn umap-learn matplotlib seaborn tqdm pillow

echo "----------------------------------------------------------"
echo "[4/4] Instalasi Selesai!"
echo "----------------------------------------------------------"
echo "Untuk mulai bekerja, jalankan perintah:"
echo "  source dino-env/bin/activate"
echo ""
echo "Silakan baca PANDUAN_MENJALANKAN.md untuk instruksi selanjutnya."
echo "----------------------------------------------------------"
