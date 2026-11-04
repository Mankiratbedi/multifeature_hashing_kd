
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Teacher(nn.Module):
    """Pretrained ResNet-18 used as an image feature extractor."""

    def __init__(self):
        super().__init__()

        # Load ResNet-18 with pretrained ImageNet weights.
        self.backbone = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        # Remove the final 1000-class classification layer.
        self.backbone.fc = nn.Identity()

        # Freeze parameters for baseline feature extraction.
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def forward(self, images):
        """Return one 512-dimensional feature vector per image."""
        return self.backbone(images)


if __name__ == "__main__":
    model = ResNet18Teacher()
    model.eval()

    # Test using a batch of two RGB images resized to 224 x 224.
    dummy_images = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        features = model(dummy_images)

    print("Input shape:", dummy_images.shape)
    print("Feature shape:", features.shape)
    print("Feature extraction successful!")