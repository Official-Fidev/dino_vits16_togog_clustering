Oke ini bahan diskusi lengkap semuanya — jurnal + Bab 1, 2, dan 3!

---

## 📋 BAHAN DISKUSI LENGKAP

---

## 📰 BAGIAN 1 — JURNAL

### ✅ 6 Jurnal STAY
| No | Judul | Jurnal |
|----|-------|--------|
| 1 | A Comprehensive Survey of Image Clustering Based on Deep Learning | Pattern Recognition – ScienceDirect |
| 2 | Deep Image Clustering: A Survey | Neurocomputing – ScienceDirect |
| 3 | Rethinking Deep Clustering: Self-Supervision Is All You Need | Neural Networks – ScienceDirect |
| 4 | Fusion ViT + UMAP Framework for Image Clustering | Expert Systems – ScienceDirect |
| 5 | Deep EM Network for Unsupervised Clustering | Image and Vision Computing – ScienceDirect |
| 6 (baru) | Towards Unsupervised Domain-Specific Open-World Recognition **(ViT + HDBSCAN)** | Neurocomputing – ScienceDirect |

### ❌ 4 Jurnal yang Masih Perlu Dicari Penggantinya (No. 7, 8, 9, 10)
> Topik yang dicari: **ViT-S16 feature extraction**, **UMAP dimensionality reduction**, **HDBSCAN density clustering**, **unsupervised image classification** — semua harus ScienceDirect

---

## 📝 BAGIAN 2 — BAB 1

### Yang STAY di Bab 1
| Sub-bab | Status | Alasan |
|---------|--------|--------|
| 1.1 Latar Belakang (paragraf 1–3) | ✅ Stay | Konteks budaya Bali + masalah klasifikasi manual tetap relevan |
| 1.3 Tujuan Penelitian (poin 1–3) | ✅ Stay struktur | Hanya ganti nama metode |
| 1.4 Manfaat Penelitian | ✅ Stay | Tidak ada kaitan langsung dengan metode |

### Yang BERUBAH di Bab 1
| Sub-bab | Kalimat/Bagian yang Berubah | Lama | Baru |
|---------|---------------------------|------|------|
| 1.1 Latar Belakang paragraf 4 | Nama model | "DINO (Self-Distillation with No Labels)" | "Vision Transformer (ViT-S16)" |
| 1.1 Latar Belakang paragraf 5 | Spesifikasi model | "akurasi k-NN 74,5%, fitur 384 dimensi" | "fitur **768 dimensi**, patch 16x16" |
| 1.1 Latar Belakang paragraf 6 | Nama metode clustering | "K-Means + Silhouette Score" | "HDBSCAN + jumlah cluster otomatis" |
| 1.2 Rumusan Masalah poin 1 | Nama metode | "K-Means + DINO ViT-S16" | "ViT-S16 + UMAP + HDBSCAN" |
| 1.2 Rumusan Masalah poin 3 | Evaluasi | "Silhouette Score" | "DBCV Score + Silhouette Score (non-outlier)" |
| 1.3 Tujuan poin 3 | Evaluasi | "Silhouette Score" | "DBCV Score + Silhouette Score" |
| 1.5 Batasan poin 2 | Nama model | "dino_vits16" | "ViT-S16 (vit_small_patch16_224)" |
| 1.5 Batasan poin 3 | Metode | "K-Means + Silhouette Score" | "HDBSCAN, jumlah cluster ditentukan otomatis oleh algoritma" |
| 1.5 Batasan poin 4 | Reduksi dimensi | tidak ada UMAP | Tambah: "UMAP untuk reduksi dimensi ke 2D" |

**Estimasi perubahan Bab 1: ~10 kalimat yang diedit, tidak ada yang ditulis ulang total**

---

## 📚 BAGIAN 3 — BAB 2

### Yang STAY di Bab 2
| Sub-bab | Status | Alasan |
|---------|--------|--------|
| 2.1 Tabel Penelitian Terdahulu | ⚠️ Update 5 baris | Ganti jurnal No. 6–10 sesuai pengganti |
| 2.2 Patung Tradisional Bali | ✅ Stay | Tidak ada kaitan dengan metode |
| 2.3 Unsupervised Clustering (intro) | ✅ Stay | Konsep umum tetap relevan |
| 2.4 Self-Supervised Learning | ✅ Stay | Masih relevan, ViT-S16 tetap self-supervised |
| 2.5 Vision Transformer (ViT) | ✅ Stay | Justru makin relevan karena metode utamanya ViT |
| 2.8 Google Colab & VS Code | ✅ Stay | Tools tidak berubah |
| 2.9 Kerangka Pikir | ⚠️ Revisi ringan | Ganti nama metode di narasi |

### Yang BERUBAH di Bab 2
| Sub-bab Lama | Nasib | Diganti Jadi |
|-------------|-------|-------------|
| 2.3.1 K-Means Clustering | ❌ Hapus | **2.3.1 HDBSCAN** (Density-Based Clustering) |
| 2.3.2 Silhouette Score | ⚠️ Pindah posisi | Tetap ada tapi jadi **2.3.3 Evaluasi: DBCV Score & Silhouette Score** |
| 2.6 Model DINO ViT-S16 | ⚠️ Revisi judul & isi | Jadi **2.6 Model ViT-S16** — hilangkan narasi DINO, fokus ke arsitektur ViT-S16, output **768 dimensi** |
| 2.7 PCA | ❌ Hapus | **2.7 UMAP** (Uniform Manifold Approximation and Projection) |

**Estimasi perubahan Bab 2: 3 sub-bab ditulis ulang (2.3.1, 2.6, 2.7), 1 sub-bab pindah posisi**

---

## 🔬 BAGIAN 4 — BAB 3 (BARU, BELUM ADA)

Bab 3 akan ditulis dari nol. Ini strukturnya:

### Struktur Bab 3 yang Diusulkan

```
BAB III — METODE PENELITIAN

3.1 Jenis Penelitian
3.2 Waktu dan Tempat Penelitian
3.3 Alat dan Bahan
    3.3.1 Perangkat Keras
    3.3.2 Perangkat Lunak
3.4 Dataset
    → Dataset patung Bali (PatungBali.v1i.coco)
    → Jumlah gambar, format, sumber
3.5 Alur Penelitian (mengacu flowchart yang ada)
3.6 Tahapan Penelitian
    3.6.1 Preprocessing
          → Resize 256x256, normalisasi piksel, simpan manifest.csv
    3.6.2 Ekstraksi Fitur dengan ViT-S16
          → Inisialisasi model, cek GPU/CPU
          → Proses batch via Transformer Encoder
          → Output: vektor 768 dimensi → simpan features.npy
    3.6.3 Reduksi Dimensi dengan UMAP
          → Input: fitur 768 dimensi
          → Parameter: n_components=2, metric=cosine/euclidean
          → Output: embeddings_umap_2d.npy
    3.6.4 Clustering dengan HDBSCAN
          → Input: embeddings_umap_2d.npy
          → Parameter: min_cluster_size (minimal, tidak mempengaruhi jumlah cluster)
          → Jumlah cluster ditentukan otomatis oleh HDBSCAN
          → Dense region → ID cluster (0, 1, 2, ...)
          → Sparse/isolated → outlier (-1)
          → Output: cluster_assignments.csv
    3.6.5 Evaluasi
          → DBCV Score (validitas kepadatan cluster)
          → Silhouette Score (hanya pada data non-outlier)
    3.6.6 Visualisasi & Pelaporan
          → Scatter plot 2D per cluster
          → Comparison grid gambar per cluster
          → Export: hdbscan_result.png + results/summary.md
3.7 Diagram Alur Penelitian
    → Gambar 3.1: Overall Pipeline
    → Gambar 3.2: Detail HDBSCAN
    → Gambar 3.3: Visualisasi & Reporting
    → Gambar 3.4: Detail Preprocessing
    → Gambar 3.5: Detail Ekstraksi Fitur ViT-S16
    → Gambar 3.6: Detail UMAP
```

---

## ✅ RINGKASAN KEPUTUSAN YANG PERLU DIKONFIRMASI TIM

| No | Pertanyaan | Opsi |
|----|-----------|------|
| 1 | Jurnal No. 7, 8, 9, 10 — sudah ada kandidat atau mau aku carikan lagi? | Minta carikan / Tim sudah punya |
| 2 | Fitur ViT-S16 = **768 dimensi** (sesuai flowchart gambar)? | ✅ Konfirmasi / ❌ Koreksi |
| 3 | Bab 3 pakai struktur di atas atau ada tambahan dari kampus? | ✅ Pakai ini / Ada format kampus |
| 4 | Gambar flowchart di Bab 3 — mau dimasukkan ke docx atau cukup deskripsi naratifnya? | Masukkan gambar / Naratif saja |

---

Kalau semua udah dikonfirmasi tim, tinggal bilang **"gas ubah semua"** dan aku langsung kerjain revisi Bab 1 & 2 + tulis Bab 3 dalam satu file docx! 🔥