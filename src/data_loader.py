import torch
import torchvision as tv
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import logging
import argparse

class DinoDataset:
    """Dataset loader and preprocessor for DINO ViT-S16"""

    def __init__(self, input_dir, output_dir, manifest_path=None):
        """
        Initialize the dataset loader.

        Args:
            input_dir: Directory containing resized images (256x256)
            output_dir: Directory to save processed tensors (224x224)
            manifest_path: Optional CSV file with metadata
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.transform = tv.transforms.Compose([
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]    # ImageNet std
            )
        ])

        # Collect image files
        self.image_files = []
        extensions = ['.png', '.jpg', '.jpeg', '.JPEG', '.JPG']

        for ext in extensions:
            self.image_files.extend(list(self.input_dir.glob(f"*{ext}")))

        # Load manifest if provided
        self.manifest = {}
        if manifest_path:
            self.load_manifest(manifest_path)

        logging.info(f"Found {len(self.image_files)} images in {input_dir}")

    def load_manifest(self, manifest_path):
        """Load manifest CSV with metadata"""
        import csv

        with open(manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.manifest[row['filename']] = row

    def process_and_save(self):
        """
        Process all images and save as tensors.
        Returns statistics about the processing.
        """
        stats = {
            'total_images': len(self.image_files),
            'processed': 0,
            'failed': 0
        }

        logging.info("Starting preprocessing...")

        for img_path in tqdm(self.image_files, desc="Processing images"):
            try:
                # Load image
                image = Image.open(img_path).convert('RGB')

                # Apply transforms
                tensor = self.transform(image)

                # Save tensor
                output_path = self.output_dir / f"{img_path.stem}.pt"
                torch.save(tensor, output_path)

                stats['processed'] += 1

            except Exception as e:
                logging.error(f"Failed to process {img_path}: {str(e)}")
                stats['failed'] += 1

        logging.info(f"Processing complete. Success: {stats['processed']}, Failed: {stats['failed']}")
        return stats

    def create_manifest(self, output_path):
        """Create a manifest CSV with all processed files"""
        manifest_data = []

        for pt_file in self.output_dir.glob("*.pt"):
            manifest_data.append({
                'filename': pt_file.name,
                'original_path': str(pt_file).replace(str(self.output_dir), str(self.input_dir)).replace('.pt', Path(pt_file.stem).suffix),
                'processed_path': str(pt_file),
                'size': pt_file.stat().st_size
            })

        import csv

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=manifest_data[0].keys())
            writer.writeheader()
            writer.writerows(manifest_data)

        logging.info(f"Manifest saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess images for DINO ViT-S16')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing resized images')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save processed tensors')
    parser.add_argument('--manifest', type=str, default=None,
                       help='Optional manifest CSV file')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Create dataset processor
    dataset = DinoDataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest
    )

    # Process images
    stats = dataset.process_and_save()

    # Create manifest
    manifest_path = Path(args.output_dir) / 'manifest.csv'
    dataset.create_manifest(str(manifest_path))

    print(f"\nPreprocessing complete!")
    print(f"Total images: {stats['total_images']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Manifest: {manifest_path}")

if __name__ == "__main__":
    main()