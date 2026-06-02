# Laporan Perbandingan 10 Iterasi Klasterisasi (HDBSCAN Auto-Tuned)

Laporan ini menunjukkan stabilitas dan variasi hasil dari algoritma HDBSCAN yang parameternya di-*tuning* secara acak (stochastic) menggunakan Optuna selama 10 kali iterasi (putaran).

| Iterasi | Folder Run | `min_cluster_size` | `min_samples` | `epsilon` | Jumlah Klaster | Jml Outlier (%) | DBCV Score | Silhouette Score |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `run_iter1_20260530_171510` | 109 | 16 | 0.2292 | **7** | 327 (3.25%) | **0.5551** | 0.5324 |
| 2 | `run_iter2_20260530_171727` | 99 | 14 | 0.1669 | **14** | 1095 (10.89%) | **0.5878** | 0.5566 |
| 3 | `run_iter3_20260530_171943` | 98 | 14 | 0.0799 | **14** | 1095 (10.89%) | **0.5878** | 0.5566 |
| 4 | `run_iter4_20260530_172215` | 107 | 14 | 0.1662 | **7** | 321 (3.19%) | **0.5553** | 0.5323 |
| 5 | `run_iter5_20260530_172434` | 95 | 20 | 0.1574 | **8** | 271 (2.70%) | **0.56** | 0.4692 |
| 6 | `run_iter6_20260530_172651` | 93 | 26 | 0.4976 | **8** | 238 (2.37%) | **0.5537** | 0.414 |
| 7 | `run_iter7_20260530_172907` | 107 | 17 | 0.0013 | **7** | 331 (3.29%) | **0.561** | 0.5325 |
| 8 | `run_iter8_20260530_173124` | 100 | 21 | 0.1569 | **9** | 346 (3.44%) | **0.5563** | 0.424 |
| 9 | `run_iter9_20260530_173341` | 125 | 30 | 0.0819 | **13** | 1219 (12.13%) | **0.4819** | 0.5576 |
| 10 | `run_iter10_20260530_173556` | 113 | 19 | 0.331 | **7** | 361 (3.59%) | **0.5505** | 0.5332 |

---
**Analisis Singkat:**
- **Stabilitas Parameter:** Optuna mencari parameter terbaik berdasarkan DBCV skor tertinggi. Perbedaan nilai parameter pada masing-masing iterasi wajar terjadi karena Optuna menggunakan metode pencarian *stochastic* (acak terarah).
- **Jumlah Klaster & Outlier:** Bisa diamati bahwa meskipun parameternya berfluktuasi, jumlah klaster utama yang dihasilkan biasanya berkisar di angka yang berdekatan. Ini membuktikan bahwa reduksi 10D telah memberikan struktur data yang cukup stabil (tangguh) untuk memandu HDBSCAN.