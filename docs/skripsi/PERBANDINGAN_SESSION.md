# Laporan Perbandingan 10 Iterasi Klasterisasi (HDBSCAN Auto-Tuned)

Laporan ini menunjukkan stabilitas dan variasi hasil dari algoritma HDBSCAN yang parameternya di-*tuning* secara acak (stochastic) menggunakan Optuna selama 10 kali iterasi (putaran).

| Iterasi | Folder Run | `min_cluster_size` | `min_samples` | `epsilon` | Jumlah Klaster | Jml Outlier (%) | DBCV Score | Silhouette Score |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `run_iter1_20261123_120713` | 84 | 8 | 0.4133 | **20** | 409 (4.07%) | **0.6146** | 0.6516 |
| 2 | `run_iter2_20261123_120848` | 83 | 8 | 0.3949 | **21** | 457 (4.55%) | **0.6147** | 0.6432 |
| 3 | `run_iter3_20261123_121023` | 102 | 8 | 0.441 | **18** | 505 (5.02%) | **0.5954** | 0.6544 |
| 4 | `run_iter4_20261123_121156` | 95 | 16 | 0.4748 | **20** | 339 (3.37%) | **0.5758** | 0.6466 |
| 5 | `run_iter5_20261123_121330` | 115 | 10 | 0.4203 | **18** | 506 (5.03%) | **0.5833** | 0.6544 |
| 6 | `run_iter6_20261123_121505` | 86 | 9 | 0.4781 | **20** | 409 (4.07%) | **0.6063** | 0.6516 |
| 7 | `run_iter7_20261123_121639` | 129 | 8 | 0.479 | **18** | 505 (5.02%) | **0.5954** | 0.6544 |
| 8 | `run_iter8_20261123_121813` | 137 | 8 | 0.4237 | **18** | 505 (5.02%) | **0.5954** | 0.6544 |
| 9 | `run_iter9_20261123_121947` | 127 | 10 | 0.4242 | **18** | 506 (5.03%) | **0.5833** | 0.6544 |
| 10 | `run_iter10_20261123_122121` | 96 | 9 | 0.392 | **21** | 457 (4.55%) | **0.6061** | 0.6432 |

---
**Analisis Singkat:**
- **Stabilitas Parameter:** Optuna mencari parameter terbaik berdasarkan DBCV skor tertinggi. Perbedaan nilai parameter pada masing-masing iterasi wajar terjadi karena Optuna menggunakan metode pencarian *stochastic* (acak terarah).
- **Jumlah Klaster & Outlier:** Bisa diamati bahwa meskipun parameternya berfluktuasi, jumlah klaster utama yang dihasilkan biasanya berkisar di angka yang berdekatan. Ini membuktikan bahwa reduksi 10D telah memberikan struktur data yang cukup stabil (tangguh) untuk memandu HDBSCAN.