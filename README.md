# DINO ViT-S16 Unsupervised Clustering for Balinese Statues

Repositori ini berisi implementasi sistem klasifikasi visual otomatis untuk patung tradisional Bali menggunakan model **DINO ViT-S16** dan algoritma **Unsupervised Clustering**. Project ini mampu mengelompokkan ribuan citra patung berdasarkan kemiripan visual tanpa memerlukan label manual.

## 🚀 Fitur Utama
- **Self-Supervised Learning**: Menggunakan backbone DINO ViT-S16 untuk ekstraksi fitur yang kaya.
- **Deep Feature Extraction**: Transformasi gambar menjadi vektor fitur 768-dimensi.
- **Efficient Dimensionality Reduction**: Reduksi dimensi menggunakan UMAP untuk performa clustering yang lebih baik.
- **Automated Clustering**: Pengelompokan otomatis dengan K-Means.
- **Interactive Setup**: Skrip instalasi cerdas yang mendeteksi dukungan GPU/CUDA secara otomatis.

---

## 🛠️ Persiapan & Instalasi

Project ini dilengkapi dengan skrip instalasi otomatis untuk memudahkan konfigurasi environment.

### 1. Kloning Repositori
```bash
git clone https://github.com/Official-Fidev/dino_vits16_togog_clustering.git
cd dino_vits16_togog_clustering
```

### 2. Jalankan Auto-Setup
Skrip ini akan membuat *virtual environment* dan menginstal PyTorch sesuai hardware Anda (CPU atau GPU).
```bash
chmod +x setup.sh
./setup.sh
```

### 3. Aktivasi Environment
Selalu aktifkan environment sebelum menjalankan program:
```bash
source dino-env/bin/activate
```

---

## 📂 Alur Kerja (Pipeline)

Anda dapat menjalankan seluruh proses secara otomatis atau langkah demi langkah.

### A. Jalankan Seluruh Proses (Otomatis)
```bash
chmod +x run_all.sh
./run_all.sh
```

### B. Jalankan Langkah demi Langkah (Manual)

1.  **Preprocessing**: Menyiapkan dataset gambar.
    ```bash
    python3 preprocess.py
    ```
2.  **Feature Extraction**: Mengekstrak ciri visual menggunakan DINO.
    ```bash
    python3 extract_features.py
    ```
3.  **Dimensionality Reduction**: Reduksi fitur ke 2D menggunakan UMAP.
    ```bash
    python3 src/dim_reducer.py --method umap --n-components 2
    ```
4.  **Clustering**: Mengelompokkan gambar ke dalam 8 cluster (default).
    ```bash
    python3 src/clustering.py --method kmeans --n-clusters 8 --save-plots
    ```
5.  **Visualization**: Menghasilkan scatter plot dan grid contoh gambar.
    ```bash
    python3 visualize_clusters.py
    ```

---

## 📊 Struktur Folder
- `src/`: Modul utama (loader, model, clustering, dll).
- `data/`: Dataset gambar (tidak masuk Git).
- `features/`: Vektor fitur hasil ekstraksi.
- `embeddings/`: Hasil reduksi dimensi (UMAP/PCA).
- `plots/`: Hasil visualisasi grafik dan sampel cluster.
- `results/summary.md`: Laporan performa clustering.

---

## 📖 Dokumentasi Tambahan
- [Panduan Menjalankan](PANDUAN_MENJALANKAN.md): Detail parameter dan instruksi manual.
- [Flowcharts](FLOWCHARTS.md): Diagram alur logika sistem.

---
*Developed for research on Balinese Traditional Statue Classification.*
