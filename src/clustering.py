import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import hdbscan
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import argparse
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os


class ClusteringEngine:
    """Clustering engine for reduced embeddings"""

    def __init__(self, method='kmeans', n_clusters=8, **kwargs):
        """
        Initialize clustering engine.

        Args:
            method: Clustering method ('kmeans', 'dbscan', 'agglomerative', 'hdbscan')
            n_clusters: Number of clusters (for methods that require it)
            **kwargs: Additional parameters for the method
        """
        self.method = method
        self.n_clusters = n_clusters
        self.clusterer = None
        self.labels_ = None
        self.params = kwargs

        # Initialize clusterer based on method
        if method == 'kmeans':
            kmeans_kwargs = {k: v for k, v in kwargs.items() if k in ['init', 'n_init', 'max_iter', 'tol']}
            self.clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init='auto' if 'n_init' not in kmeans_kwargs else kmeans_kwargs['n_init'],
                **kmeans_kwargs
            )
        elif method == 'dbscan':
            dbscan_kwargs = {k: v for k, v in kwargs.items() if k in ['eps', 'min_samples', 'metric', 'algorithm']}
            self.clusterer = DBSCAN(**dbscan_kwargs)
        elif method == 'hdbscan':
            hdb_kwargs = {k: v for k, v in kwargs.items() if k in ['min_cluster_size', 'min_samples', 'cluster_selection_method', 'metric']}
            self.clusterer = hdbscan.HDBSCAN(
                prediction_data=True,
                gen_min_span_tree=True,
                **hdb_kwargs
            )
        elif method == 'agglomerative':
            agg_kwargs = {k: v for k, v in kwargs.items() if k in ['metric', 'linkage']}
            self.clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                **agg_kwargs
            )
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        logging.info(f"Initialized {method} clustering")

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit clusterer and predict clusters.

        Args:
            embeddings: Input embeddings (n_samples, n_features)

        Returns:
            Cluster labels
        """
        logging.info(f"Clustering {embeddings.shape[0]} samples using {self.method}")

        self.labels_ = self.clusterer.fit_predict(embeddings)

        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = list(self.labels_).count(-1)

        logging.info(f"Found {n_clusters} clusters")
        if n_noise > 0:
            logging.info(f"Found {n_noise} noise points (outliers)")

        return self.labels_

    def get_cluster_stats(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Get clustering statistics"""
        if self.labels_ is None:
            raise ValueError("Not clustered yet")

        mask = self.labels_ != -1
        n_clusters = len(set(self.labels_[mask]))
        n_noise = int((self.labels_ == -1).sum())
        
        stats = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'outlier_ratio_pct': round(n_noise / len(self.labels_) * 100, 2),
            'cluster_sizes': pd.Series(self.labels_).value_counts().to_dict(),
            'method': self.method,
            'params': self.params
        }

        # Calculate evaluation metrics (excluding noise points)
        if n_clusters >= 2:
            if np.sum(mask) >= n_clusters + 1:
                try:
                    stats['silhouette_score'] = round(float(silhouette_score(
                        embeddings[mask], self.labels_[mask]
                    )), 4)
                    stats['davies_bouldin_score'] = round(float(davies_bouldin_score(
                        embeddings[mask], self.labels_[mask]
                    )), 4)
                    
                    if self.method == 'hdbscan':
                        stats['dbcv_score'] = round(float(self.clusterer.relative_validity_), 4)
                except Exception as e:
                    logging.warning(f"Could not calculate metrics: {e}")

        return stats

    def save_labels(self, labels: np.ndarray, output_path: str, filenames: Optional[List[str]] = None):
        """Save cluster labels to CSV"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df_data = {'cluster': labels}
        if filenames:
            df_data['filename'] = filenames

        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False)
        logging.info(f"Cluster assignments saved to {output_path}")

        return df


def plot_hdbscan_result(embeddings, labels, stats, output_dir):
    """Plot HDBSCAN results with scatter and bar chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    unique_labels = sorted(set(labels))
    palette = plt.cm.tab10.colors

    # Scatter plot
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            ax1.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                       c='lightgray', s=15, alpha=0.4, label='Outlier', zorder=1)
        else:
            color = palette[label % len(palette)]
            ax1.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                       c=[color], s=40, alpha=0.75, label=f'Cluster {label}', zorder=2)
    
    title = f"HDBSCAN Result - {stats['n_clusters']} clusters"
    if 'dbcv_score' in stats:
        title += f"\nDBCV: {stats['dbcv_score']}"
    ax1.set_title(title)
    ax1.set_xlabel("UMAP Dim 1")
    ax1.set_ylabel("UMAP Dim 2")
    ax1.legend(loc='best', fontsize=8)

    # Bar chart
    cluster_labels = [l for l in unique_labels if l != -1]
    counts = [int((labels == l).sum()) for l in cluster_labels]
    colors = [palette[l % len(palette)] for l in cluster_labels]
    
    ax2.bar([f"C{l}" for l in cluster_labels], counts, color=colors)
    if -1 in unique_labels:
        ax2.bar(["Outlier"], [int((labels == -1).sum())], color='lightgray')
    ax2.set_title("Cluster Sizes")
    ax2.set_ylabel("Number of Images")

    plt.tight_layout()
    # Path output plot diperbaiki agar ke root plots/
    plot_path = Path("plots") / "hdbscan_result.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logging.info(f"HDBSCAN plot saved to {plot_path}")


def cluster_embeddings(
    embeddings_path: str,
    output_dir: str,
    method: str = 'kmeans',
    n_clusters: int = 8,
    save_plots: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """Cluster embeddings entry point"""
    embeddings = np.load(embeddings_path)
    logging.info(f"Loaded embeddings with shape: {embeddings.shape}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    clusterer = ClusteringEngine(method=method, n_clusters=n_clusters, **kwargs)
    labels = clusterer.fit_predict(embeddings)
    stats = clusterer.get_cluster_stats(embeddings)

    labels_path = output_path / f"cluster_assignments_{method}.csv"
    df = clusterer.save_labels(labels, str(labels_path))

    if method == 'hdbscan' and save_plots:
        plot_hdbscan_result(embeddings, labels, stats, output_dir)

    stats_path = output_path / f"cluster_stats_{method}.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    return {
        'labels': labels,
        'stats': stats,
        'df': df,
        'labels_path': str(labels_path),
        'stats_path': str(stats_path)
    }


def main():
    parser = argparse.ArgumentParser(description='Cluster embeddings')
    parser.add_argument('--embeddings', type=str, required=True, help='Path to embeddings.npy')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--method', type=str, default='kmeans', choices=['kmeans', 'dbscan', 'hdbscan', 'agglomerative'])
    parser.add_argument('--n-clusters', type=int, default=8, help='K for KMeans')
    parser.add_argument('--min-cluster-size', type=int, default=5, help='HDBSCAN parameter')
    parser.add_argument('--min-samples', type=int, default=None, help='HDBSCAN parameter')
    parser.add_argument('--save-plots', action='store_true')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    results = cluster_embeddings(
        embeddings_path=args.embeddings,
        output_dir=args.output_dir,
        method=args.method,
        n_clusters=args.n_clusters,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        save_plots=args.save_plots
    )

    print(f"\nClustering complete via {args.method}!")
    print(f"Clusters found: {results['stats']['n_clusters']}")
    if args.method == 'hdbscan':
        print(f"Outliers: {results['stats']['n_noise']} ({results['stats']['outlier_ratio_pct']}%)")
        if 'dbcv_score' in results['stats']:
            print(f"DBCV Score: {results['stats']['dbcv_score']}")
    if 'silhouette_score' in results['stats']:
        print(f"Silhouette Score: {results['stats']['silhouette_score']}")


if __name__ == "__main__":
    main()