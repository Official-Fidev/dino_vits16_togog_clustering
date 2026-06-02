import optuna
import numpy as np
import hdbscan
import json
import logging
from pathlib import Path
import argparse
import sys

def objective(trial, embeddings):
    mcs = trial.suggest_int('min_cluster_size', 10, 150)
    ms = trial.suggest_int('min_samples', 5, 30)
    eps = trial.suggest_float('cluster_selection_epsilon', 0.0, 0.5)

    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            cluster_selection_epsilon=eps,
            metric='euclidean',
            gen_min_span_tree=True
        )
        labels = clusterer.fit_predict(embeddings)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # We want to maximize DBCV. If clusters < 2, it's a failure.
        if n_clusters < 2:
            return -1.0
            
        dbcv_score = clusterer.relative_validity_
        
        # Penalize overly high outliers (optional, e.g., if outlier ratio > 40%)
        outlier_ratio = list(labels).count(-1) / len(labels)
        if outlier_ratio > 0.4:
            return dbcv_score - (outlier_ratio * 0.5)
            
        return dbcv_score
    except Exception as e:
        return -1.0

def main():
    parser = argparse.ArgumentParser(description='Auto-tune HDBSCAN with Optuna')
    parser.add_argument('--embeddings', type=str, required=True, help='Path to 10D embeddings')
    parser.add_argument('--output', type=str, default='outputs/results/best_params.json')
    parser.add_argument('--n-trials', type=int, default=30)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    if not Path(args.embeddings).exists():
        logging.error(f"Embeddings file not found: {args.embeddings}")
        sys.exit(1)
        
    logging.info(f"Loading embeddings from {args.embeddings}")
    embeddings = np.load(args.embeddings)
    
    # Suppress noisy optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction='maximize')
    
    logging.info(f"Starting Optuna search for {args.n_trials} trials...")
    study.optimize(lambda trial: objective(trial, embeddings), n_trials=args.n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    best_value = study.best_value
    
    logging.info(f"\nOptimization finished!")
    logging.info(f"Best DBCV: {best_value:.4f}")
    logging.info(f"Best params: {best_params}")
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(best_params, f, indent=2)
        
    logging.info(f"Best parameters written to {args.output}")

if __name__ == '__main__':
    main()
