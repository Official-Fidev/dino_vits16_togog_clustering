import numpy as np
import pandas as pd
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
    n_samples: int = 5
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
    
    clusters = sorted(df['cluster'].unique())
    n_clusters = len(clusters)
    
    # Create grid plot
    fig, axes = plt.subplots(n_clusters, n_samples, figsize=(n_samples * 3, n_clusters * 3))
    fig.suptitle('Cluster Comparison (Rows: Clusters, Columns: Samples)', fontsize=20)
    
    for row, cluster_id in enumerate(clusters):
        # Get samples for this cluster
        cluster_df = df[df['cluster'] == cluster_id]
        samples = cluster_df.sample(min(n_samples, len(cluster_df)), random_state=42)
        
        for col in range(n_samples):
            ax = axes[row, col] if n_clusters > 1 else axes[col]
            
            if col < len(samples):
                pt_filename = samples.iloc[col]['filename']
                img_filename = pt_filename.replace('.pt', '.png')
                img_path = Path(raw_images_dir) / img_filename
                
                # Try finding image
                if not img_path.exists():
                    for ext in ['.jpg', '.jpeg']:
                        alt_path = Path(raw_images_dir) / pt_filename.replace('.pt', ext)
                        if alt_path.exists():
                            img_path = alt_path
                            break
                
                if img_path.exists():
                    img = Image.open(img_path)
                    ax.imshow(img)
                    if col == 0:
                        ax.set_ylabel(f"Cluster {cluster_id}", fontsize=14, fontweight='bold')
                else:
                    ax.text(0.5, 0.5, "Not Found", ha='center', va='center')
            
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    comparison_path = output_path / "cluster_comparison_grid.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
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
    
    args = parser.parse_args()
    
    create_cluster_comparison(
        clusters_path=args.clusters,
        metadata_path=args.metadata,
        raw_images_dir=args.raw_images,
        output_dir=args.output_dir,
        n_samples=args.samples
    )
