# Teks Skrip: Penjelasan Pipeline "DINO ViT-S16 HDBSCAN Clustering"

Berikut adalah teks narasi lengkap yang dirancang untuk presentasi atau *dubbing* video dokumentasi. Teks ini telah diperbarui untuk merefleksikan penggunaan arsitektur **HDBSCAN** yang lebih cerdas dan murni otomatis.

---

### [1. Pembukaan]
"Hari ini saya akan menjelaskan alur kerja dari sistem *Unsupervised Clustering* untuk dataset patung tradisional Bali. Berbeda dengan sistem lawas yang menebak-nebak jumlah kelompok, sistem kita sekarang menggunakan algoritma **HDBSCAN**. 

Algoritma ini membaca 'kepadatan' data untuk menentukan jumlah kelompok secara murni otomatis, persis seperti bagaimana mata manusia melihat sekumpulan bintang di langit. Mari kita bedah alurnya langkah demi langkah."

---

### [2. Preprocessing & Feature Extraction]
"Tahap pertama adalah menyiapkan gambar. File `preprocess.py` memotong gambar patung tepat di tengah menjadi ukuran 224x224 piksel agar fokus pada objek utama.

Lalu, kita menggunakan AI mutakhir buatan Meta, yaitu **DINO ViT-S16**, lewat file `extract_features.py`. Model DINO bertugas mengekstrak ciri-ciri visual patung—seperti ukiran, bentuk wajah, atau mahkota—dan mengubahnya menjadi DNA visual berbentuk vektor 768-dimensi."

---

### [3. Dimensionality Reduction (UMAP)]
"Karena vektor 768-dimensi terlalu rumit, kita memampatkannya menggunakan algoritma **UMAP**. UMAP menyederhanakan data menjadi hanya 2 dimensi (X dan Y) seperti peta datar, namun tetap menjaga agar patung-patung yang DNA visualnya mirip tetap berada berdekatan di peta tersebut."

---

### [4. Clustering (HDBSCAN - Otomatis)]
"Nah, di sinilah keunggulan utama sistem ini. File `src/clustering.py` menjalankan algoritma **HDBSCAN**. Kita tidak perlu lagi mengatur berapa jumlah klaster yang kita inginkan. 

Sistem akan mencari area-area yang padat di peta UMAP. Area padat ini otomatis diangkat menjadi sebuah kelompok (Cluster). Hebatnya lagi, gambar-gambar patung yang aneh, buram, atau tidak mirip dengan kelompok manapun tidak akan dipaksa masuk. HDBSCAN akan menandai mereka sebagai **Outlier** atau 'Data Anomali' dengan label -1. Ini membuat kelompok utama kita tetap sangat murni dan akurat."

---

### [5. Evaluasi Kualitas (DBCV & Silhouette)]
"Untuk menilai kualitas pengelompokan secara ilmiah, sistem menghitung metrik **DBCV (Density-Based Cluster Validity)**. Metrik ini khusus dibuat untuk HDBSCAN, yang menilai seberapa padat isi sebuah klaster dan seberapa kosong jarak antar klaster. 

Sebagai pendukung, sistem juga menghitung **Silhouette Score**, tetapi secara cerdas mengabaikan titik-titik anomali (Outlier) agar nilainya tidak rusak."

---

### [6. Visualisasi & Penutup]
"Sebagai bukti akhir, sistem membuat visualisasi canggih di folder `plots/` berupa `hdbscan_result.png`. Di sana kita bisa melihat bar chart jumlah anggota klaster, beserta Scatter Plot di mana titik-titik abu-abu transparan menunjukkan patung-patung yang terdeteksi sebagai anomali.

Kesimpulannya, dengan kombinasi DINO dan HDBSCAN, kita telah menciptakan sistem klasifikasi budaya Bali yang sepenuhnya mandiri, cerdas mengenali bentuk, dan tahan terhadap gambar-gambar anomali."

---

### 💡 Tips Presentasi:
*   **Sorot Outlier:** Tunjukkan pada audiens beberapa gambar yang masuk ke kategori Outlier (label -1). Buktikan bahwa gambar-gambar tersebut memang memiliki sudut pandang yang aneh, kualitas buruk, atau jenis patung yang sangat langka di dataset. Ini adalah *"selling point"* HDBSCAN.
*   **Fokus pada Otomatisasi:** Tekankan bahwa tidak ada campur tangan manusia (seperti `--n-clusters 8`) di sini. Data itu sendiri yang berbicara berapa kelompok yang wajar untuk dibentuk.
