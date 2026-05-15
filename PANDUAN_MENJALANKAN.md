# Panduan Menjalankan Project: DINO Clustering

Dokumen ini menjelaskan cara menjalankan tahapan pipeline clustering untuk citra patung tradisional Bali.

## 1. Persiapan Lingkungan (Setup)

Gunakan skrip otomatis untuk membuat *virtual environment* dan menginstal semua library yang dibutuhkan:

```bash
chmod +x setup.sh
./setup.sh
```

Skrip ini akan mendeteksi apakah komputer Anda memiliki GPU (NVIDIA) dan menginstal versi PyTorch yang sesuai secara otomatis.

---

## 2. Aktivasi Environment

Sebelum menjalankan perintah apapun, pastikan *environment* sudah aktif:

```bash
source dino-env/bin/activate
```

---

## 3. Tahapan Pipeline

Project ini memiliki beberapa skrip utama yang bisa dijalankan secara terpisah atau sekaligus.

### A. Menjalankan Seluruh Pipeline (Otomatis)
Gunakan skrip utama untuk menjalankan dari reduksi dimensi hingga visualisasi:
```bash
chmod +x run_all.sh
./run_all.sh
```

### B. Menjalankan Skrip Individual (Manual)

#### 1. Preprocessing Data
Menyiapkan gambar agar siap diolah oleh model (resize & normalisasi).
```bash
python3 preprocess.py
```

#### 2. Ekstraksi Fitur (Feature Extraction)
Menggunakan model DINO ViT-S16 untuk mengambil ciri unik dari gambar.
```bash
python3 extract_features.py
```

#### 3. Reduksi Dimensi (Dimensionality Reduction)
Mengubah fitur 768-dimensi menjadi 2-dimensi (UMAP) agar bisa di-cluster.
```bash
python3 src/dim_reducer.py --method umap --n-components 2
```

#### 4. Clustering (K-Means)
Mengelompokkan data berdasarkan kemiripan visual.
```bash
python3 src/clustering.py --method kmeans --n-clusters 8 --save-plots
```

#### 5. Visualisasi
Melihat hasil clustering dalam bentuk grafik scatter plot dan contoh gambar per cluster.
```bash
python3 visualize_clusters.py
```

---

## 4. Parameter Penting

Beberapa skrip mendukung parameter tambahan untuk kustomisasi:

| Parameter | Deskripsi | Default |
|-----------|-----------|---------|
| `--n-clusters` | Jumlah kelompok (K) pada K-Means | `8` |
| `--method` | Metode reduksi (`umap`, `pca`, `tsne`) | `umap` |
| `--samples` | Jumlah sampel gambar per cluster yang ditampilkan | `15` |
| `--output-dir` | Folder tempat menyimpan hasil | (Sesuai tahap) |

---

## 5. Lokasi Hasil
- **Embeddings**: `embeddings/` (Hasil reduksi dimensi)
- **Clusters**: `clusters/` (Hasil pelabelan kelompok)
- **Plots**: `plots/` (Visualisasi grafik & grid contoh gambar)
- **Summary**: `results/summary.md` (Laporan performa clustering)
