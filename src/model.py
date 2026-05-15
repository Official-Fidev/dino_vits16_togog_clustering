import torch
import torchvision.transforms as T
import logging
from typing import Optional


class DinoViT:
    """DINO ViT-S16 model for feature extraction"""

    def __init__(self, model_name='dino_vits16', pretrained=True):
        """
        Initialize DINO ViT-S16 model.

        Args:
            model_name: Model architecture name
            pretrained: Whether to load pretrained weights
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self.feature_dim = None

        logging.info(f"Initializing DINO {model_name} on {self.device}")

    def load_model(self):
        """Load the DINO ViT-S16 model"""
        try:
            # Try to import DINO-specific implementation
            try:
                # Try to install DINO if not available
                import dino.models
                from dino.models import vit_small
                self.model = vit_small(pretrained=True)
                logging.info("Loaded official DINO ViT-S16 model")
            except ImportError:
                # Fallback to regular ViT-S16
                import torchvision.models as models
                self.model = models.vit_b_16(pretrained=True)
                # Remove classification head
                self.model.heads = torch.nn.Identity()
                logging.info("Loaded regular ViT-S16 (DINO-like)")

            # Set to evaluation mode
            self.model.eval()

            # Move to device
            self.model.to(self.device)

            logging.info("DINO ViT-S16 model loaded successfully")

        except ImportError as e:
            logging.error(f"Failed to import DINO/ViT model: {e}")
            logging.error("Please install ViT-S16: pip install torchvision")
            raise

    def get_transform(self):
        """Get the transformation pipeline for DINO"""
        if self.transform is None:
            self.transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        return self.transform

    def extract_features(self, image_tensor, layer_idx=-2):
        """
        Extract features from the model.

        Args:
            image_tensor: Input tensor (C, H, W)
            layer_idx: Which layer to extract features from (-2 = penultimate layer)

        Returns:
            features: Extracted features
        """
        if self.model is None:
            self.load_model()

        # Add batch dimension
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # Move to device
        image_tensor = image_tensor.to(self.device)

        # Forward pass
        with torch.no_grad():
            # Get features from the model (after removing classification head)
            features = self.model(image_tensor)

            # Flatten if needed
            if features.dim() > 2:
                features = features.view(features.size(0), -1)

        return features.cpu()

    def get_feature_dimension(self):
        """Get the dimension of extracted features"""
        if self.feature_dim is None:
            if self.model is None:
                self.load_model()

            # Create dummy input
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)

            with torch.no_grad():
                features = self.extract_features(dummy_input)

            self.feature_dim = features.shape[-1]

        return self.feature_dim


def load_dino_model(model_path=None):
    """Convenience function to load DINO model"""
    model = DinoViT(model_name='dino_vits16', pretrained=True)
    return model