import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import argparse
import json
from typing import List, Dict

from .model import DinoViT
from .data_loader import DinoDataset


class FeatureExtractor:
    """Feature extractor using DINO ViT-S16"""

    def __init__(self, model_name='dino_vits16', device='auto'):
        """
        Initialize feature extractor.

        Args:
            model_name: DINO model architecture
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name

        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize model
        self.dino_model = DinoViT(model_name='vit_s16', pretrained=True)
        self.dino_model.load_model()

        logging.info(f"Feature extractor initialized on {self.device}")

    def extract_from_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract features from a single tensor.

        Args:
            tensor: Input tensor (C, H, W)

        Returns:
            features: Extracted features
        """
        # Move to device
        tensor = tensor.to(self.device)

        # Extract features
        with torch.no_grad():
            # Add batch dimension
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)

            # Forward pass through DINO
            features = self.dino_model.model(tensor)

            # Get penultimate layer features (before classification)
            features = features.view(features.size(0), -1)  # Flatten

        return features.cpu()

    def extract_from_directory(self, input_dir: str, output_dir: str, batch_size: int = 32):
        """
        Extract features from all tensors in a directory.

        Args:
            input_dir: Directory containing .pt files
            output_dir: Directory to save features
            batch_size: Batch size for processing

        Returns:
            stats: Extraction statistics
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all tensor files
        tensor_files = list(input_path.glob("*.pt"))
        logging.info(f"Found {len(tensor_files)} tensor files")

        # Process in batches
        features_list = []
        filenames = []
        stats = {
            'total_files': len(tensor_files),
            'processed': 0,
            'failed': 0
        }

        for i in tqdm(range(0, len(tensor_files), batch_size), desc="Extracting features"):
            batch_files = tensor_files[i:i + batch_size]
            batch_tensors = []

            # Load batch
            for pt_file in batch_files:
                try:
                    tensor = torch.load(pt_file)
                    batch_tensors.append(tensor)
                    filenames.append(pt_file.name)
                    stats['processed'] += 1
                except Exception as e:
                    logging.error(f"Failed to load {pt_file}: {str(e)}")
                    stats['failed'] += 1

            if batch_tensors:
                # Stack tensors into batch
                batch = torch.stack(batch_tensors).to(self.device)

                # Extract features
                with torch.no_grad():
                    features = self.dino_model.model(batch)
                    features = features.view(batch.size(0), -1)  # Flatten

                features_list.append(features.cpu())

        # Concatenate all features
        if features_list:
            all_features = torch.cat(features_list, dim=0)
            logging.info(f"Extracted features shape: {all_features.shape}")

            # Save features
            features_path = output_path / "features.npy"
            np.save(features_path, all_features.numpy())

            # Save feature metadata
            metadata = {
                'feature_dim': all_features.shape[1],
                'num_samples': all_features.shape[0],
                'model_name': self.model_name,
                'filenames': filenames
            }

            metadata_path = output_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logging.info(f"Features saved to {features_path}")
            logging.info(f"Metadata saved to {metadata_path}")

        return stats


def main():
    parser = argparse.ArgumentParser(description='Extract features using DINO ViT-S16')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing processed .pt files')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save extracted features')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for processing')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for extraction')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Create feature extractor
    extractor = FeatureExtractor(
        model_name='dino_vits16',
        device=args.device
    )

    # Extract features
    logging.info("Starting feature extraction...")
    stats = extractor.extract_from_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )

    print(f"\nFeature extraction complete!")
    print(f"Total files: {stats['total_files']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Failed: {stats['failed']}")


if __name__ == "__main__":
    main()