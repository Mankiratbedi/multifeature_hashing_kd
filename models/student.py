
import torch
import torch.nn as nn


class StudentCNN(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()

        # Extract visual features from an image
        self.features = nn.Sequential(
            # Block 1: 3 channels -> 32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 2: 32 channels -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 3: 64 channels -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 4: 128 channels -> 256 channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Convert spatial feature maps to one vector
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Convert features to the required embedding dimension
        self.projection = nn.Linear(256, feature_dim)

    def forward(self, images):
        features = self.features(images)
        embeddings = self.projection(features)
        return embeddings


if __name__ == "__main__":
    model = StudentCNN(feature_dim=256)

    # Test using two CIFAR-10-sized images
    dummy_images = torch.randn(2, 3, 32, 32)

    embeddings = model(dummy_images)

    print("Input shape:", dummy_images.shape)
    print("Output shape:", embeddings.shape)
    print("Student CNN test successful!")
