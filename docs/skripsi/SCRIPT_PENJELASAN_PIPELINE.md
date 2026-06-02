# Teks Skrip: Penjelasan Pipeline "DINO ViT-S16 HDBSCAN Clustering"

Berikut adalah teks narasi yang sudah disesuaikan dengan **gaya bahasamu** (lebih santai, *to-the-point*, dan menjelaskan alur logika per langkah) namun tetap sopan untuk presentasi sidang skripsi.

---

### [Tahap 1: Pra-pemrosesan]
"Tahap pertama adalah pra-pemrosesan. Yang kita lakukan di sini adalah mengambil gambar-gambar mentah patung, terus kita lakukan *center crop* agar ukurannya seragam menjadi 224x224 piksel. Setelah ukurannya sama, gambar ini kita konversi menjadi format tensor dan dinormalisasi warnanya biar siap dibaca oleh model AI."

---

### [Tahap 2: Ekstraksi Fitur (DINO ViT-S16)]
"Sekarang kita masuk ke tahap kedua, yaitu proses ekstraksi fitur. Yang dilakukan di tahap ini adalah kita *load* model **DINO ViT-S16** dulu. Model ini sifatnya *self-supervised* (bisa belajar sendiri tanpa perlu dilabeli manual). 

Kemudian, file tensor gambar 224x224 yang kita punya tadi akan dimasukkan ke model, lalu dipecah jadi potongan-potongan kecil berukuran 16x16 piksel. Nah, masing-masing potongan gambar ini diberikan penanda (*attention*) oleh DINO ViT-S16 untuk menangkap ciri khas patung tersebut. Setelah diproses, seluruh informasi ini dikonversi menjadi array berukuran **768 dimensi**."

---

### [Tahap 3 & 5: Reduksi Dimensi (10D & 2D)]
"Selanjutnya kita masuk ke proses reduksi dimensi. Array 768 dimensi tadi kan terlalu besar dan ribet buat dihitung oleh mesin, jadi kita perkecil pakai algoritma **UMAP** menjadi **10 dimensi (10D)** saja. Ini penting supaya proses pencarian klasternya nanti lebih akurat dan optimal.

Bersamaan dengan itu (di Tahap 5), array 768 dimensi yang asli juga kita reduksi lagi jadi **2 dimensi (2D)**. Tapi ingat, yang 2D ini *murni* cuma dipakai untuk menggambar titik *X* dan *Y* di grafik visualisasi layar nanti, bukan untuk dihitung sama algoritma klasterisasinya."

---

### [Tahap 4: Klasterisasi (Optuna & HDBSCAN)]
"Masuk ke tahap keempat yaitu klasterisasi. Di sini kita menggunakan algoritma **HDBSCAN**. Tapi sebelum HDBSCAN jalan, kita eksekusi **Optuna** dulu. Optuna ini kita tugaskan untuk melakukan *tuning* secara otomatis guna mencari nilai parameter yang paling bagus.

Setelah parameter terbaiknya ketemu, barulah HDBSCAN kita *fit* ke data 10 dimensi tadi. Keunggulan HDBSCAN ini adalah dia bisa otomatis mencari area data yang padat untuk dijadikan klaster, sekaligus membuang gambar-gambar patung yang tidak jelas atau *blur* menjadi **Outlier** (diberi label -1)."

---

### [Tahap 6: Evaluasi & Laporan Akhir]
"Terakhir, di tahap keenam kita gabungkan semuanya. Titik koordinat 2D yang udah kita bikin di Tahap 5 tadi kita beri warna berdasarkan hasil klaster dari HDBSCAN. Jadilah *Scatter Plot*. 

Lalu kita hitung kualitas klasternya pakai **DBCV Score** dan **Silhouette Score** untuk melihat seberapa padat kelompok yang terbentuk. Semua grafik, skor, dan contoh gambarnya (*comparison grid*) otomatis disimpan ke laporan akhir."

---

### 💡 Tips Q&A Dosen (Bahasa Santai):
*   **Kenapa DINO?** "Karena kalau pakai CNN biasa, dia cuma lihat garis atau pola kasar. DINO (Transformer) ini pecah gambar jadi 16x16, jadi dia bisa paham secara keseluruhan, misal oh ini mahkota nyambungnya ke wajah."
*   **Kenapa ada 10D dan 2D?** "Kalau langsung di-klaster pakai 2D, data aslinya bakal rusak karena diperas terlalu ekstrem. Makanya mesinnya disuruh ngerjain yang 10D, nah yang 2D buat mata kita aja pas nampilin grafik."
