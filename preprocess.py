#!/usr/bin/env python3
"""
Preprocessing script for DINO ViT-S16
Processes resized datasets (256x256) into ViT-S16 compatible format (224x224)
"""

import sys
sys.path.append('src')

from src.data_loader import DinoDataset
import logging

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Process the resized datasets
    logging.info("Starting preprocessing of resized datasets...")

    dataset = DinoDataset(
        input_dir="data/raw/resized_datasets/resized_256x256",
        output_dir="data/processed/resized_256x256"
    )

    # Process all images
    stats = dataset.process_and_save()

    # Create manifest
    manifest_path = "data/processed/resized_256x256/manifest.csv"
    dataset.create_manifest(manifest_path)

    print("\n" + "="*50)
    print("PREPROCESSING COMPLETE!")
    print("="*50)
    print(f"Input directory: data/raw/resized_datasets/resized_256x256")
    print(f"Output directory: data/processed/resized_256x256")
    print(f"Total images found: {stats['total_images']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Manifest saved to: {manifest_path}")
    print("="*50)

if __name__ == "__main__":
    main()