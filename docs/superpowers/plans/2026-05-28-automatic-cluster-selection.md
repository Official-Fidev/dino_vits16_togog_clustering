# HDBSCAN Clustering Implementation Plan
> **Menggantikan** pendekatan `--auto-k` (K-Means + Silhouette/Davies-Bouldin) dengan **HDBSCAN** — algoritma berbasis kepadatan yang menentukan jumlah cluster secara mandiri tanpa perlu range K sama sekali.

---

## Perubahan Filosofi

| Aspek | Rencana Lama | Rencana Baru |
|---|---|---|
| Algoritma | K-Means | HDBSCAN |
| Penentu K | Range 2–12 + scoring | Otomatis dari struktur data |
| Parameter utama | `--k-range` | `--min-cluster-size` |
| Outlier | Dipaksa masuk cluster | Ditandai label `-1` |
| Bentuk cluster | Hanya bulat | Bebas (lonjong, tak beraturan) |
| Metrik validasi | Silhouette + Davies-Bouldin | DBCV (Density-Based Validity) |

---

## Task 1: Refactor `ClusteringEngine` di `src/clustering.py`

**File:** `src/clustering.py`

- [ ] **Step 1: Ganti dependency**

```python
# Hapus:
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Tambah:
import hdbscan
from sklearn.metrics import silhouette_score  # tetap dipakai untuk laporan akhir
```

- [ ] **Step 2: Hapus method `find_optimal_k`, ganti dengan `run_hdbscan`**

```python
def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = None,
) -> tuple[np.ndarray, hdbscan.HDBSCAN]:
    """
    Jalankan HDBSCAN. Tidak butuh jumlah cluster di awal.
    
    Args:
        embeddings:        Array 2D koordinat UMAP (N x 2).
        min_cluster_size:  Minimum anggota untuk dianggap 1 cluster.
                           Sesuaikan dengan ukuran dataset:
                           - Dataset kecil (<100 gambar):  3–5
                           - Dataset sedang (100–500):     5–10
                           - Dataset besar (>500):         10–20
        min_samples:       Ketatnya definisi "inti cluster".
                           None = pakai nilai min_cluster_size (rekomendasi default).

    Returns:
        labels:    Array (N,) berisi nomor cluster. Nilai -1 = outlier.
        clusterer: Objek HDBSCAN yang sudah fit (untuk akses metadata).
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',   # 'eom' lebih stabil dari 'leaf'
        prediction_data=True,             # dibutuhkan untuk soft clustering
    )
    labels = clusterer.fit_predict(embeddings)
    return labels, clusterer
```

- [ ] **Step 3: Tambah method `evaluate_hdbscan` untuk laporan**

```python
def evaluate_hdbscan(
    embeddings: np.ndarray,
    labels: np.ndarray,
    clusterer: hdbscan.HDBSCAN,
) -> dict:
    """
    Hitung metrik kualitas hasil HDBSCAN.
    Hanya dihitung pada titik NON-outlier.
    """
    mask = labels != -1
    n_clusters  = len(set(labels[mask]))
    n_outliers  = int((labels == -1).sum())
    n_total     = len(labels)

    metrics = {
        "n_clusters":        n_clusters,
        "n_outliers":        n_outliers,
        "outlier_ratio_pct": round(n_outliers / n_total * 100, 2),
    }

    # Silhouette hanya valid jika minimal ada 2 cluster dan 2+ non-outlier per cluster
    if n_clusters >= 2 and mask.sum() >= n_clusters + 1:
        metrics["silhouette_score"] = round(
            silhouette_score(embeddings[mask], labels[mask]), 4
        )
    else:
        metrics["silhouette_score"] = None

    # DBCV (Density-Based Cluster Validity) — metrik asli untuk HDBSCAN
    # Tersedia langsung dari objek clusterer
    metrics["dbcv_score"] = round(float(clusterer.relative_validity_), 4)

    return metrics
```

- [ ] **Step 4: Update method `cluster_embeddings` sebagai entry point utama**

```python
def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int = None,        # diabaikan jika use_hdbscan=True
    use_hdbscan: bool = False,
    min_cluster_size: int = 5,
    min_samples: int = None,
    output_dir: str = "clusters",
) -> pd.DataFrame:
    
    os.makedirs(output_dir, exist_ok=True)

    if use_hdbscan:
        print("[HDBSCAN] Mendeteksi cluster secara otomatis...")
        labels, clusterer = run_hdbscan(embeddings, min_cluster_size, min_samples)
        metrics = evaluate_hdbscan(embeddings, labels, clusterer)

        print(f"[HDBSCAN] Cluster ditemukan : {metrics['n_clusters']}")
        print(f"[HDBSCAN] Outlier           : {metrics['n_outliers']} gambar "
              f"({metrics['outlier_ratio_pct']}%)")
        print(f"[HDBSCAN] DBCV Score        : {metrics['dbcv_score']}")
        if metrics["silhouette_score"]:
            print(f"[HDBSCAN] Silhouette Score  : {metrics['silhouette_score']}")

        plot_hdbscan_result(embeddings, labels, metrics, output_dir)

    else:
        # Jalur lama: K-Means manual
        k = n_clusters or 8
        print(f"[KMeans] Clustering dengan K={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(embeddings)

    df = pd.DataFrame({"embedding_idx": range(len(labels)), "cluster": labels})
    df.to_csv(os.path.join(output_dir, "cluster_assignments.csv"), index=False)
    return df
```

- [ ] **Step 5: Tambah fungsi `plot_hdbscan_result` untuk visualisasi**

```python
def plot_hdbscan_result(
    embeddings: np.ndarray,
    labels: np.ndarray,
    metrics: dict,
    output_dir: str,
) -> None:
    """
    Buat dua subplot:
    - Kiri:  Scatter plot warna per cluster (outlier = abu-abu kecil)
    - Kanan: Bar chart jumlah anggota per cluster
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    unique_labels = sorted(set(labels))
    
    # Palette warna — outlier selalu abu-abu
    palette = plt.cm.tab10.colors
    
    # --- Scatter plot ---
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            ax1.scatter(
                embeddings[mask, 0], embeddings[mask, 1],
                c='lightgray', s=15, alpha=0.4, label='Outlier', zorder=1
            )
        else:
            color = palette[label % len(palette)]
            ax1.scatter(
                embeddings[mask, 0], embeddings[mask, 1],
                c=[color], s=40, alpha=0.75,
                label=f'Cluster {label} (n={mask.sum()})', zorder=2
            )
    
    ax1.set_title(
        f"HDBSCAN — {metrics['n_clusters']} cluster ditemukan\n"
        f"DBCV: {metrics['dbcv_score']}  |  "
        f"Outlier: {metrics['n_outliers']} ({metrics['outlier_ratio_pct']}%)"
    )
    ax1.set_xlabel("UMAP Dim 1")
    ax1.set_ylabel("UMAP Dim 2")
    ax1.legend(loc='best', fontsize=8)

    # --- Bar chart anggota per cluster ---
    cluster_labels = [l for l in unique_labels if l != -1]
    counts = [(labels == l).sum() for l in cluster_labels]
    colors = [palette[l % len(palette)] for l in cluster_labels]
    
    ax2.bar([f"C{l}" for l in cluster_labels], counts, color=colors, edgecolor='white')
    if -1 in unique_labels:
        ax2.bar(["Outlier"], [(labels == -1).sum()], color='lightgray', edgecolor='white')
    ax2.set_title("Jumlah anggota per cluster")
    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("Jumlah gambar")

    fig.tight_layout()
    out_path = os.path.join(output_dir, "..", "plots", "hdbscan_result.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot] Disimpan: {out_path}")
```

---

## Task 2: Update CLI dan `run_all.sh`

**File:** `src/clustering.py` (bagian `main`) dan `run_all.sh`

- [ ] **Step 1: Ganti argumen CLI**

```python
# Hapus argumen lama:
# parser.add_argument('--auto-k',   action='store_true')
# parser.add_argument('--k-range',  nargs=2, type=int, default=[2, 12])

# Tambah argumen baru:
parser.add_argument(
    '--hdbscan',
    action='store_true',
    help='Gunakan HDBSCAN untuk clustering otomatis tanpa menentukan K.'
)
parser.add_argument(
    '--min-cluster-size',
    type=int,
    default=5,
    metavar='N',
    help='Minimum anggota per cluster (default: 5). '
         'Naikkan jika dataset besar atau cluster terlalu banyak.'
)
parser.add_argument(
    '--min-samples',
    type=int,
    default=None,
    metavar='N',
    help='Keketatan inti cluster. Default: sama dengan --min-cluster-size.'
)
```

- [ ] **Step 2: Update blok `main`**

```python
if __name__ == "__main__":
    args = parser.parse_args()
    embeddings = np.load(args.embeddings_path)

    cluster_embeddings(
        embeddings=embeddings,
        n_clusters=args.n_clusters,      # diabaikan jika --hdbscan aktif
        use_hdbscan=args.hdbscan,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        output_dir=args.output_dir,
    )
```

- [ ] **Step 3: Update `run_all.sh`**

```bash
# Hapus baris lama:
# python src/clustering.py --auto-k --k-range 2 12

# Ganti dengan:
python src/clustering.py \
    --hdbscan \
    --min-cluster-size 5 \
    # Sesuaikan nilai di atas dengan ukuran dataset patung Bali kamu
```

- [ ] **Step 4: Tambah instalasi `hdbscan` ke `setup.sh`**

```bash
# Di dalam blok instalasi pip, tambah:
pip install hdbscan

# Atau jika pakai requirements.txt, tambah baris:
hdbscan>=0.8.33
```

---

## Task 3: Update Visualisasi Output

**File:** `visualize_clusters.py`

- [ ] **Step 1: Tangani label outlier (`-1`) agar tidak crash**

```python
# Sebelum membuat grid sampel gambar, filter outlier:
df_clean = df[df['cluster'] != -1]

# Untuk scatter plot, beri warna khusus pada outlier:
outlier_mask = df['cluster'] == -1
if outlier_mask.any():
    ax.scatter(
        embeddings[outlier_mask, 0],
        embeddings[outlier_mask, 1],
        c='lightgray', s=10, alpha=0.3,
        label=f'Outlier (n={outlier_mask.sum()})'
    )
```

- [ ] **Step 2: Update judul plot agar mencantumkan info HDBSCAN**

```python
n_clusters = df['cluster'].nunique() - (1 if -1 in df['cluster'].values else 0)
ax.set_title(f"HDBSCAN Clustering — {n_clusters} cluster ditemukan secara otomatis")
```

---

## Task 4: Verifikasi & Dokumentasi

- [ ] **Step 1: Jalankan pipeline dan cek log**

```bash
bash run_all.sh
# Output yang diharapkan:
# [HDBSCAN] Cluster ditemukan : X
# [HDBSCAN] Outlier           : Y gambar (Z%)
# [HDBSCAN] DBCV Score        : 0.XXXX
```

- [ ] **Step 2: Verifikasi output file**

```
plots/
├── hdbscan_result.png        ← scatter + bar chart baru
clusters/
└── cluster_assignments.csv   ← kolom 'cluster' berisi -1 untuk outlier
```

- [ ] **Step 3: Update `results/summary.md`**

Ganti bagian metrik lama dengan template ini:

```markdown
## Hasil Clustering

| Metrik              | Nilai     |
|---------------------|-----------|
| Algoritma           | HDBSCAN   |
| Cluster ditemukan   | X         |
| Total gambar        | N         |
| Outlier             | Y (Z%)    |
| DBCV Score          | 0.XXXX    |
| Silhouette Score    | 0.XXXX    |
| min_cluster_size    | 5         |

> Jumlah cluster ditentukan otomatis oleh HDBSCAN berdasarkan
> struktur kepadatan data — bukan dari range K yang ditentukan manual.
```

---

## Catatan Tuning

Jika hasil HDBSCAN kurang memuaskan, coba dua penyesuaian ini sebelum ganti algoritma:

| Masalah | Solusi |
|---|---|
| Terlalu banyak cluster kecil | Naikkan `--min-cluster-size` (mis. 5 → 10) |
| Terlalu banyak outlier (>30%) | Turunkan `--min-samples` (mis. None → 2) |
| Semua jadi 1 cluster | Turunkan `--min-cluster-size` (mis. 5 → 3) |
| Cluster tidak stabil antar run | Tambah `core_dist_n_jobs=-1` ke konstruktor HDBSCAN |

Parameter `cluster_selection_method='eom'` (Excess of Mass) yang dipakai di atas menghasilkan cluster yang lebih besar dan stabil. Jika kamu ingin cluster yang lebih banyak dan granular, ganti ke `'leaf'`.