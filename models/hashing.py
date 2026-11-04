
import torch
import torch.nn as nn


class HashingLayer(nn.Module):

    def __init__(self, input_dim=256, hash_bits=32):
        super().__init__()

        self.hash_bits = hash_bits

        # Map student features to the required hash length
        self.hash_projection = nn.Linear(
            input_dim,
            hash_bits
        )

    def forward(self, features):

        # Continuous hash representation during training
        continuous_codes = torch.tanh(
            self.hash_projection(features)
        )

        return continuous_codes

    def get_binary_codes(self, features):

        # Generate binary codes during inference
        continuous_codes = self.forward(features)

        # Convert values to -1 or +1
        binary_codes = torch.where(
            continuous_codes >= 0,
            1.0,
            -1.0
        )

        return binary_codes


if __name__ == "__main__":

    model = HashingLayer(
        input_dim=256,
        hash_bits=32
    )

    dummy_features = torch.randn(4, 256)

    continuous = model(dummy_features)

    binary = model.get_binary_codes(dummy_features)

    print("Input feature shape:", dummy_features.shape)
    print("Continuous hash shape:", continuous.shape)
    print("Binary hash shape:", binary.shape)

    print("\nSample continuous code:")
    print(continuous[0])

    print("\nSample binary code:")
    print(binary[0])

    print("\nUnique binary values:", torch.unique(binary))
    print("\nHashing layer test successful!")
