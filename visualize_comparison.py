import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Gunakan backend non-interaktif agar tidak error X11
import matplotlib.pyplot as plt
import json
from pathlib import Path
from PIL import Image
import logging

def create_cluster_comparison(
    clusters_path: str,
    metadata_path: str,
    raw_images_dir: str,
    output_dir: str,
    n_samples: int = 5,
    max_display_clusters: int = 20 # Batasi jumlah baris agar tidak crash
):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df_clusters = pd.read_csv(clusters_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    filenames = metadata['filenames']
    df = pd.DataFrame({
        'cluster': df_clusters['cluster'],
        'filename': filenames
    })
    
    # Hitung ukuran tiap cluster dan urutkan dari yang terbesar
    cluster_counts = df['cluster'].value_counts()
    
    # Ambil cluster ID yang akan ditampilkan (Top N + Outliers jika ada)
    # Kita prioritaskan cluster dengan anggota terbanyak
    top_clusters = cluster_counts.index.tolist()
    
    # Jika jumlah cluster sangat banyak, batasi
    if len(top_clusters) > max_display_clusters:
        logging.info(f"Total clusters ({len(top_clusters)}) exceeds limit. Showing top {max_display_clusters} largest clusters.")
        # Pastikan outliers (-1) tetap masuk jika ada di top N, jika tidak, kita bisa tambahkan manual nanti
        display_clusters = top_clusters[:max_display_clusters]
    else:
        display_clusters = top_clusters
        
    # Urutkan agar tampil rapi (Outlier biasanya di akhir atau awal)
    display_clusters = sorted(display_clusters)
    n_display = len(display_clusters)
    
    logging.info(f"Creating comparison grid for {n_display} clusters...")
    
    # Create grid plot
    fig, axes = plt.subplots(n_display, n_samples, figsize=(n_samples * 3, n_display * 3))
    fig.suptitle(f'Cluster Comparison - Top {n_display} Clusters\n(Rows: Clusters, Columns: Samples)', fontsize=16)
    
    for row, cluster_id in enumerate(display_clusters):
        # Get samples for this cluster
        cluster_df = df[df['cluster'] == cluster_id]
        samples = cluster_df.sample(min(n_samples, len(cluster_df)), random_state=42)
        
        for col in range(n_samples):
            # Handle single row case
            if n_display > 1:
                ax = axes[row, col]
            else:
                ax = axes[col]
            
            if col < len(samples):
                pt_filename = samples.iloc[col]['filename']
                img_filename_base = pt_filename.replace('.pt', '')
                
                img_path = None
                for ext in ['.png', '.jpg', '.jpeg']:
                    alt_path = Path(raw_images_dir) / f"{img_filename_base}{ext}"
                    if alt_path.exists():
                        img_path = alt_path
                        break
                
                if img_path:
                    try:
                        img = Image.open(img_path)
                        ax.imshow(img)
                        if col == 0:
                            count = cluster_counts[cluster_id]
                            label = f"C {cluster_id}\n(n={count})" if cluster_id != -1 else f"OUTLIERS\n(n={count})"
                            ax.set_ylabel(label, fontsize=10, fontweight='bold')
                    except Exception as e:
                        ax.text(0.5, 0.5, "Err", ha='center', va='center')
                else:
                    ax.text(0.5, 0.5, "Not Found", ha='center', va='center')
            
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    comparison_path = output_path / "cluster_comparison_grid.png"
    plt.savefig(comparison_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Comparison grid saved to {comparison_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create a grid comparison of clusters')
    parser.add_argument('--clusters', type=str, required=True)
    parser.add_argument('--metadata', type=str, required=True)
    parser.add_argument('--raw-images', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--max-clusters', type=int, default=25, help='Max rows in the grid')
    
    args = parser.parse_args()
    
    create_cluster_comparison(
        clusters_path=args.clusters,
        metadata_path=args.metadata,
        raw_images_dir=args.raw_images,
        output_dir=args.output_dir,
        n_samples=args.samples,
        max_display_clusters=args.max_clusters
    )
