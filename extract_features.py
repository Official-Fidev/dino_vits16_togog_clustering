#!/usr/bin/env python3
"""
Feature extraction script using DINO ViT-S16
Extracts features from preprocessed tensors
"""

import sys
sys.path.append('src')

from src.feature_extractor import FeatureExtractor
import logging


def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info("Starting DINO ViT-S16 feature extraction...")

    # Create feature extractor
    extractor = FeatureExtractor(
        model_name='vit_s16',
        device='auto'  # Will auto-detect CUDA
    )

    # Extract features from processed datasets
    stats = extractor.extract_from_directory(
        input_dir="data/processed/resized_256x256",
        output_dir="features/resized_256x256",
        batch_size=32
    )

    print("\n" + "="*50)
    print("FEATURE EXTRACTION COMPLETE!")
    print("="*50)
    print(f"Input directory: data/processed/resized_256x256")
    print(f"Output directory: features/resized_256x256")
    print(f"Total files: {stats['total_files']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Features saved as: features/resized_256x256/features.npy")
    print(f"Metadata: features/resized_256x256/metadata.json")
    print("="*50)


if __name__ == "__main__":
    main()