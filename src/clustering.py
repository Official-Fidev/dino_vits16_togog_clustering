import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import argparse
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns


class ClusteringEngine:
    """Clustering engine for reduced embeddings"""

    def __init__(self, method='kmeans', n_clusters=8, **kwargs):
        """
        Initialize clustering engine.

        Args:
            method: Clustering method ('kmeans', 'dbscan', 'agglomerative')
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
            # Filter kwargs for KMeans
            kmeans_kwargs = {k: v for k, v in kwargs.items() if k in ['init', 'n_init', 'max_iter', 'tol']}
            self.clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                **kmeans_kwargs
            )
        elif method == 'dbscan':
            # Filter kwargs for DBSCAN
            dbscan_kwargs = {k: v for k, v in kwargs.items() if k in ['eps', 'min_samples', 'metric', 'algorithm']}
            self.clusterer = DBSCAN(**dbscan_kwargs)
        elif method == 'agglomerative':
            # Filter kwargs for AgglomerativeClustering
            agg_kwargs = {k: v for k, v in kwargs.items() if k in ['metric', 'linkage']}
            self.clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                **agg_kwargs
            )
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        logging.info(f"Initialized {method} clustering with n_clusters={n_clusters}")

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

        # Handle noise points in DBSCAN (label = -1)
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = list(self.labels_).count(-1)

        logging.info(f"Found {n_clusters} clusters")
        if n_noise > 0:
            logging.info(f"Found {n_noise} noise points")

        return self.labels_

    def get_cluster_stats(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Get clustering statistics"""
        if self.labels_ is None:
            raise ValueError("Not clustered yet")

        stats = {
            'n_clusters': len(set(self.labels_)) - (1 if -1 in self.labels_ else 0),
            'n_noise': list(self.labels_).count(-1),
            'cluster_sizes': pd.Series(self.labels_).value_counts().to_dict(),
            'method': self.method,
            'params': self.params
        }

        # Calculate evaluation metrics (excluding noise points)
        if len(set(self.labels_)) > 1:
            mask = self.labels_ != -1
            if np.sum(mask) > 1:  # Need at least 2 points for metrics
                try:
                    stats['silhouette_score'] = silhouette_score(
                        embeddings[mask], self.labels_[mask]
                    )
                    stats['calinski_harabasz_score'] = calinski_harabasz_score(
                        embeddings[mask], self.labels_[mask]
                    )
                    stats['davies_bouldin_score'] = davies_bouldin_score(
                        embeddings[mask], self.labels_[mask]
                    )
                except Exception as e:
                    logging.warning(f"Could not calculate metrics: {e}")

        return stats

    def save_labels(self, labels: np.ndarray, output_path: str, filenames: Optional[List[str]] = None):
        """Save cluster labels to CSV"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create DataFrame
        df_data = {'cluster': labels}
        if filenames:
            df_data['filename'] = filenames

        df = pd.DataFrame(df_data)

        # Save to CSV
        df.to_csv(output_path, index=False)
        logging.info(f"Cluster assignments saved to {output_path}")

        return df

    def find_optimal_k(self, embeddings: np.ndarray, k_range: range = range(2, 11)) -> Dict[int, float]:
        """
        Find optimal number of clusters using elbow method.

        Args:
            embeddings: Input embeddings
            k_range: Range of k values to try

        Returns:
            Dictionary of k values and their inertias
        """
        inertias = {}
        silhouette_scores = {}

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            inertias[k] = kmeans.inertia_

            if k > 1:
                mask = labels != -1
                if np.sum(mask) > 1:
                    silhouette_scores[k] = silhouette_score(embeddings[mask], labels[mask])

        return {
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_k': max(silhouette_scores.keys(), key=silhouette_scores.get) if silhouette_scores else None
        }


def cluster_embeddings(
    embeddings_path: str,
    output_dir: str,
    method: str = 'kmeans',
    n_clusters: int = 8,
    save_plots: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Cluster embeddings.

    Args:
        embeddings_path: Path to embeddings.npy
        output_dir: Directory to save cluster assignments
        method: Clustering method
        n_clusters: Number of clusters
        save_plots: Whether to save visualization plots
        **kwargs: Additional parameters

    Returns:
        Dictionary with clustering results
    """
    # Load embeddings
    embeddings = np.load(embeddings_path)
    logging.info(f"Loaded embeddings with shape: {embeddings.shape}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize clustering engine
    clusterer = ClusteringEngine(method=method, n_clusters=n_clusters, **kwargs)

    # Cluster embeddings
    labels = clusterer.fit_predict(embeddings)

    # Get cluster statistics
    stats = clusterer.get_cluster_stats(embeddings)

    # Save cluster assignments
    labels_path = output_path / f"cluster_assignments_{method}_{n_clusters}clusters.csv"
    df = clusterer.save_labels(labels, str(labels_path))

    # Find optimal k if using kmeans
    if method == 'kmeans':
        optimal_k_results = clusterer.find_optimal_k(embeddings)
        stats['optimal_k'] = optimal_k_results['optimal_k']
        stats['inertias'] = optimal_k_results['inertias']
        stats['silhouette_scores'] = optimal_k_results['silhouette_scores']

        # Save elbow plot
        if save_plots:
            plt.figure(figsize=(12, 5))

            # Elbow plot
            plt.subplot(1, 2, 1)
            plt.plot(list(stats['inertias'].keys()), list(stats['inertias'].values()), 'bo-')
            plt.xlabel('Number of clusters (k)')
            plt.ylabel('Inertia')
            plt.title('Elbow Method')
            plt.grid(True)

            # Silhouette plot
            plt.subplot(1, 2, 2)
            if stats['silhouette_scores']:
                plt.plot(list(stats['silhouette_scores'].keys()),
                        list(stats['silhouette_scores'].values()), 'go-')
                plt.xlabel('Number of clusters (k)')
                plt.ylabel('Silhouette Score')
                plt.title('Silhouette Score')
                plt.grid(True)

            plt.tight_layout()
            plot_path = output_path / f"elbow_silhouette_{method}_{n_clusters}clusters.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Elbow plot saved to {plot_path}")

    # Save statistics
    stats_path = output_path / f"cluster_stats_{method}_{n_clusters}clusters.json"
    with open(stats_path, 'w') as f:
        import json
        json.dump(stats, f, indent=2)

    logging.info(f"Clustering statistics saved to {stats_path}")

    return {
        'labels': labels,
        'stats': stats,
        'df': df,
        'labels_path': str(labels_path),
        'stats_path': str(stats_path)
    }


def main():
    parser = argparse.ArgumentParser(description='Cluster embeddings')
    parser.add_argument('--embeddings', type=str, required=True,
                       help='Path to embeddings.npy')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save cluster assignments')
    parser.add_argument('--method', type=str, default='kmeans',
                       choices=['kmeans', 'dbscan', 'agglomerative'],
                       help='Clustering method')
    parser.add_argument('--n-clusters', type=int, default=8,
                       help='Number of clusters (for kmeans/agglomerative)')
    parser.add_argument('--eps', type=float, default=0.5,
                       help='Epsilon for DBSCAN (default: 0.5)')
    parser.add_argument('--min-samples', type=int, default=5,
                       help='Minimum samples for DBSCAN (default: 5)')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save elbow and silhouette plots')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Cluster embeddings
    results = cluster_embeddings(
        embeddings_path=args.embeddings,
        output_dir=args.output_dir,
        method=args.method,
        n_clusters=args.n_clusters,
        eps=args.eps,
        min_samples=args.min_samples,
        save_plots=args.save_plots
    )

    print(f"\nClustering complete!")
    print(f"Method: {results['stats']['method']}")
    print(f"Number of clusters: {results['stats']['n_clusters']}")
    if 'optimal_k' in results['stats']:
        print(f"Optimal k (silhouette): {results['stats']['optimal_k']}")
    if 'silhouette_score' in results['stats']:
        print(f"Silhouette score: {results['stats']['silhouette_score']:.4f}")
    print(f"Cluster assignments saved to: {results['labels_path']}")


if __name__ == "__main__":
    main()