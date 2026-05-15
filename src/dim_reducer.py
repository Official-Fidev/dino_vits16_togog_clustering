import torch
import numpy as np
import umap
import pickle
import logging
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse
from typing import Dict, Any, Optional


class DimensionalityReducer:
    """Dimensionality reduction for extracted features"""

    def __init__(self, method='umap', n_components=2, **kwargs):
        """
        Initialize dimensionality reducer.

        Args:
            method: Reduction method ('pca', 'umap', 'tsne')
            n_components: Number of dimensions to reduce to
            **kwargs: Additional parameters for the method
        """
        self.method = method
        self.n_components = n_components
        self.reducer = None
        self.params = kwargs

        # Initialize reducer based on method
        if method == 'pca':
            self.reducer = PCA(n_components=n_components, **kwargs)
        elif method == 'umap':
            # Filter kwargs for UMAP
            umap_kwargs = {k: v for k, v in kwargs.items() if k in ['n_neighbors', 'min_dist', 'metric', 'learning_rate']}
            self.reducer = umap.UMAP(
                n_components=n_components,
                random_state=42,
                **umap_kwargs
            )
        elif method == 'tsne':
            # Filter kwargs for t-SNE
            tsne_kwargs = {k: v for k, v in kwargs.items() if k in ['perplexity', 'early_exaggeration', 'learning_rate', 'n_iter']}
            self.reducer = TSNE(
                n_components=n_components,
                random_state=42,
                **tsne_kwargs
            )
        else:
            raise ValueError(f"Unknown reduction method: {method}")

        logging.info(f"Initialized {method} reducer with {n_components} components")

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Fit reducer and transform features.

        Args:
            features: Input features (n_samples, n_features)

        Returns:
            Reduced features (n_samples, n_components)
        """
        logging.info(f"Reducing {features.shape[1]}D to {self.n_components}D using {self.method}")

        if self.method == 'tsne':
            # t-SNE doesn't support transform after fit, so we return the result directly
            reduced_features = self.reducer.fit_transform(features)
        else:
            reduced_features = self.reducer.fit_transform(features)

        logging.info(f"Reduced features shape: {reduced_features.shape}")
        return reduced_features

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted reducer.

        Args:
            features: Input features (n_samples, n_features)

        Returns:
            Reduced features (n_samples, n_components)
        """
        if self.reducer is None:
            raise ValueError("Reducer not fitted yet")

        return self.reducer.transform(features)

    def save(self, save_path: str):
        """Save the fitted reducer"""
        if self.reducer is None:
            raise ValueError("Reducer not fitted yet")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump(self.reducer, f)

        logging.info(f"Reducer saved to {save_path}")

    def load(self, load_path: str):
        """Load a fitted reducer"""
        load_path = Path(load_path)

        with open(load_path, 'rb') as f:
            self.reducer = pickle.load(f)

        logging.info(f"Reducer loaded from {load_path}")

    def get_reduction_params(self) -> Dict[str, Any]:
        """Get reduction parameters"""
        return {
            'method': self.method,
            'n_components': self.n_components,
            'params': self.params
        }


def reduce_features(
    features_path: str,
    output_dir: str,
    method: str = 'umap',
    n_components: int = 2,
    **kwargs
) -> Dict[str, Any]:
    """
    Reduce dimensionality of features.

    Args:
        features_path: Path to features.npy
        output_dir: Directory to save reduced features
        method: Reduction method
        n_components: Number of components
        **kwargs: Additional parameters

    Returns:
        Dictionary with reduction results
    """
    # Load features
    features = np.load(features_path)
    logging.info(f"Loaded features with shape: {features.shape}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize reducer
    reducer = DimensionalityReducer(method=method, n_components=n_components, **kwargs)

    # Reduce dimensions
    reduced_features = reducer.fit_transform(features)

    # Save reduced features
    reduced_path = output_path / f"embeddings_{method}_{n_components}d.npy"
    np.save(reduced_path, reduced_features)

    # Save reducer
    reducer_path = output_path / f"reducer_{method}_{n_components}d.pkl"
    reducer.save(str(reducer_path))

    # Save metadata
    metadata = {
        'original_shape': features.shape,
        'reduced_shape': reduced_features.shape,
        'method': method,
        'n_components': n_components,
        'params': kwargs,
        'reducer_path': str(reducer_path),
        'embeddings_path': str(reduced_path)
    }

    metadata_path = output_path / f"metadata_{method}_{n_components}d.json"
    with open(metadata_path, 'w') as f:
        import json
        json.dump(metadata, f, indent=2)

    logging.info(f"Reduced features saved to {reduced_path}")
    logging.info(f"Metadata saved to {metadata_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description='Reduce dimensionality of features')
    parser.add_argument('--features', type=str, required=True,
                       help='Path to features.npy')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save reduced features')
    parser.add_argument('--method', type=str, default='umap',
                       choices=['pca', 'umap', 'tsne'],
                       help='Dimensionality reduction method')
    parser.add_argument('--n-components', type=int, default=2,
                       help='Number of dimensions to reduce to')
    parser.add_argument('--n-neighbors', type=int, default=15,
                       help='Number of neighbors for UMAP (default: 15)')
    parser.add_argument('--min-dist', type=float, default=0.1,
                       help='Minimum distance for UMAP (default: 0.1)')
    parser.add_argument('--perplexity', type=int, default=30,
                       help='Perplexity for t-SNE (default: 30)')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Reduce features
    metadata = reduce_features(
        features_path=args.features,
        output_dir=args.output_dir,
        method=args.method,
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        perplexity=args.perplexity
    )

    print(f"\nDimensionality reduction complete!")
    print(f"Method: {metadata['method']}")
    print(f"Original shape: {metadata['original_shape']}")
    print(f"Reduced shape: {metadata['reduced_shape']}")
    print(f"Reduced features saved to: {metadata['embeddings_path']}")


if __name__ == "__main__":
    main()