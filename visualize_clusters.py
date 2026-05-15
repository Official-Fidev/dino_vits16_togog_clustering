import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import shutil
from pathlib import Path
from PIL import Image
import logging

def visualize_clusters(
    embeddings_path: str,
    clusters_path: str,
    metadata_path: str,
    output_dir: str,
    raw_images_dir: str,
    n_samples_per_cluster: int = 10
):
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logging.info("Loading data...")
    embeddings = np.load(embeddings_path)
    df_clusters = pd.read_csv(clusters_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    filenames = metadata['filenames']
    
    # Ensure lengths match
    if len(embeddings) != len(df_clusters) or len(embeddings) != len(filenames):
        logging.error(f"Length mismatch: embeddings({len(embeddings)}), clusters({len(df_clusters)}), filenames({len(filenames)})")
        return

    # Create visualization dataframe
    df = pd.DataFrame({
        'x': embeddings[:, 0],
        'y': embeddings[:, 1],
        'cluster': df_clusters['cluster'],
        'filename': filenames
    })
    
    # 1. Plot scatter plot
    logging.info("Generating scatter plot...")
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df, x='x', y='y', hue='cluster', palette='viridis', legend='full', alpha=0.6)
    plt.title(f"Cluster Visualization (UMAP 2D) - {len(df['cluster'].unique())} Clusters")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_path = output_path / "cluster_scatter_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Scatter plot saved to {plot_path}")
    
    # 2. Save representative samples
    logging.info(f"Saving {n_samples_per_cluster} samples per cluster...")
    samples_dir = output_path / "cluster_samples"
    if samples_dir.exists():
        shutil.rmtree(samples_dir)
    samples_dir.mkdir(parents=True)
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_samples = df[df['cluster'] == cluster_id].sample(min(n_samples_per_cluster, len(df[df['cluster'] == cluster_id])), random_state=42)
        
        cluster_subdir = samples_dir / f"cluster_{cluster_id}"
        cluster_subdir.mkdir(parents=True)
        
        for i, row in cluster_samples.iterrows():
            # Extract original image name from .pt filename
            # Metadata filenames are like "Dewi Gangga19_side_08.pt"
            pt_filename = row['filename']
            img_filename = pt_filename.replace('.pt', '.png') # Assuming original was .png
            
            src_path = Path(raw_images_dir) / img_filename
            dst_path = cluster_subdir / img_filename
            
            if src_path.exists():
                shutil.copy(src_path, dst_path)
            else:
                # Try to find the file if extension is different or nested
                found = False
                for ext in ['.png', '.jpg', '.jpeg']:
                    alt_path = Path(raw_images_dir) / pt_filename.replace('.pt', ext)
                    if alt_path.exists():
                        shutil.copy(alt_path, dst_path)
                        found = True
                        break
                if not found:
                    logging.warning(f"Could not find source image for {pt_filename} in {raw_images_dir}")

    logging.info(f"Samples saved to {samples_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize clusters and save samples')
    parser.add_argument('--embeddings', type=str, required=True)
    parser.add_argument('--clusters', type=str, required=True)
    parser.add_argument('--metadata', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--raw-images', type=str, required=True)
    parser.add_argument('--samples', type=int, default=10)
    
    args = parser.parse_args()
    
    visualize_clusters(
        embeddings_path=args.embeddings,
        clusters_path=args.clusters,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        raw_images_dir=args.raw_images,
        n_samples_per_cluster=args.samples
    )
