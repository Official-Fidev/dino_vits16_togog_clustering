# DINO ViT-S16 Unsupervised Clustering for Balinese Statues

Repositori ini berisi implementasi sistem klasifikasi visual otomatis untuk patung tradisional Bali menggunakan model **DINO ViT-S16** dan algoritma **Unsupervised Clustering**. Project ini mampu mengelompokkan ribuan citra patung berdasarkan kemiripan visual tanpa memerlukan label manual.

## 🚀 Fitur Utama
- **Self-Supervised Learning**: Menggunakan backbone DINO ViT-S16 untuk ekstraksi fitur yang kaya.
- **Deep Feature Extraction**: Transformasi gambar menjadi vektor fitur 768-dimensi.
- **Efficient Dimensionality Reduction**: Reduksi dimensi menggunakan UMAP untuk performa clustering yang lebih baik.
- **Automated Clustering**: Pengelompokan otomatis dengan HDBSCAN / K-Means.
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
Gunakan skrip utama untuk menjalankan dari reduksi dimensi hingga visualisasi:
```bash
chmod +x run_all.sh
./run_all.sh
```
Untuk pengguna Windows dengan Conda:
```cmd
run_all.bat
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
    python3 src/dim_reducer.py \
      --features outputs/features/resized_256x256/features.npy \
      --output-dir outputs/embeddings/resized_256x256 \
      --method umap \
      --n-components 2
    ```
4.  **Clustering (HDBSCAN/K-Means)**: Mengelompokkan data berdasarkan kemiripan visual.
    ```bash
    python3 src/clustering.py \
      --embeddings outputs/embeddings/resized_256x256/embeddings_umap_2d.npy \
      --output-dir outputs/clusters/resized_256x256 \
      --method hdbscan \
      --min-cluster-size 5 \
      --save-plots
    ```
5.  **Visualization**: Menghasilkan scatter plot dan grid contoh gambar.
    ```bash
    python3 visualize_clusters.py \
      --embeddings outputs/embeddings/resized_256x256/embeddings_umap_2d.npy \
      --clusters outputs/clusters/resized_256x256/cluster_assignments_hdbscan.csv \
      --metadata outputs/features/resized_256x256/metadata.json \
      --output-dir outputs/plots/resized_256x256 \
      --raw-images data/raw/resized_datasets/resized_256x256 \
      --samples 15

    python3 visualize_comparison.py \
      --clusters outputs/clusters/resized_256x256/cluster_assignments_hdbscan.csv \
      --metadata outputs/features/resized_256x256/metadata.json \
      --raw-images data/raw/resized_datasets/resized_256x256 \
      --output-dir outputs/plots/resized_256x256 \
      --samples 6
    ```

---

## ⚙️ Parameter Penting

Beberapa skrip mendukung parameter tambahan untuk kustomisasi:

| Parameter | Deskripsi | Default |
|-----------|-----------|---------|
| `--n-clusters` | Jumlah kelompok (K) pada K-Means | `8` |
| `--min-cluster-size` | Ukuran cluster minimum untuk HDBSCAN | `5` |
| `--method` | Metode reduksi (`umap`, `pca`, `tsne`) | `umap` |
| `--samples` | Jumlah sampel gambar per cluster yang ditampilkan | `15` |
| `--output-dir` | Folder tempat menyimpan hasil | (Sesuai tahap) |

---

## 📊 Struktur Folder Utama
- `src/`: Modul inti (loader, model, clustering logic, dll).
- `docs/`: Dokumen pedoman, penjelasan flowchart, dan referensi.
- `data/`: Dataset gambar (tidak masuk Git).
- `outputs/`: Folder kumpulan hasil eksekusi program.
  - `outputs/features/`: Vektor fitur hasil ekstraksi.
  - `outputs/embeddings/`: Hasil reduksi dimensi (UMAP/PCA).
  - `outputs/clusters/`: Hasil pelabelan kelompok.
  - `outputs/plots/`: Hasil visualisasi grafik dan sampel cluster.
  - `outputs/results/summary.md`: Laporan performa clustering.

---

## 📖 Dokumentasi Tambahan
- [Flowcharts](FLOWCHART_UPDATED.md): Diagram alur logika sistem terbaru.
- Untuk dokumen penjelasan teknis skripsi, periksa folder `docs/skripsi/`.

---
*Developed for research on Balinese Traditional Statue Classification.*
