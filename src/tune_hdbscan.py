import numpy as np
import hdbscan
from sklearn.metrics import silhouette_score
import itertools
import json
from pathlib import Path
import argparse
import logging

def main():
    parser = argparse.ArgumentParser(description='Tune HDBSCAN parameters')
    parser.add_argument('--embeddings', type=str, required=True, help='Path to 10D embeddings')
    parser.add_argument('--output', type=str, default='outputs/results/hdbscan_tuning.json')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    logging.info(f"Loading embeddings from {args.embeddings}")
    embeddings = np.load(args.embeddings)
    
    min_cluster_sizes = [30, 50, 80, 100]
    min_samples_list = [5, 10, 15]
    epsilons = [0.0, 0.1, 0.2, 0.5]
    
    results = []
    
    total_iters = len(min_cluster_sizes) * len(min_samples_list) * len(epsilons)
    logging.info(f"Starting grid search over {total_iters} combinations...")
    
    for mcs, ms, eps in itertools.product(min_cluster_sizes, min_samples_list, epsilons):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            cluster_selection_epsilon=eps,
            metric='euclidean',
            gen_min_span_tree=True
        )
        
        labels = clusterer.fit_predict(embeddings)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        outlier_ratio = n_noise / len(labels)
        
        sil_score = -1.0
        dbcv_score = -1.0
        
        if n_clusters >= 2:
            mask = labels != -1
            if np.sum(mask) >= n_clusters + 1:
                try:
                    sil_score = silhouette_score(embeddings[mask], labels[mask])
                    dbcv_score = clusterer.relative_validity_
                except Exception:
                    pass
                    
        result = {
            'min_cluster_size': mcs,
            'min_samples': ms,
            'cluster_selection_epsilon': eps,
            'n_clusters': n_clusters,
            'outlier_ratio': round(outlier_ratio, 4),
            'silhouette_score': round(float(sil_score), 4) if sil_score != -1 else -1,
            'dbcv_score': round(float(dbcv_score), 4) if dbcv_score != -1 else -1
        }
        results.append(result)
        logging.info(f"mcs={mcs}, ms={ms}, eps={eps} -> Clusters: {n_clusters}, Outliers: {outlier_ratio:.1%}, DBCV: {dbcv_score:.4f}, Sil: {sil_score:.4f}")

    # Sort by DBCV score descending
    results.sort(key=lambda x: x['dbcv_score'], reverse=True)
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    logging.info(f"\nTop 3 Configurations based on DBCV:")
    for i, res in enumerate(results[:3]):
        logging.info(f"{i+1}. mcs={res['min_cluster_size']}, ms={res['min_samples']}, eps={res['cluster_selection_epsilon']} -> "
                     f"Clusters: {res['n_clusters']}, Outliers: {res['outlier_ratio']:.1%}, DBCV: {res['dbcv_score']}, Sil: {res['silhouette_score']}")

if __name__ == '__main__':
    main()
