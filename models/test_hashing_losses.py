
import torch

from models.hashing_losses import total_hashing_loss


def main():

    batch_size = 8
    feature_dim = 256
    hash_bits = 32
    num_classes = 10

    # Simulated student features
    student_features = torch.randn(
        batch_size,
        feature_dim,
        requires_grad=True,
    )

    # Simulated teacher features after projection
    projected_teacher = torch.randn(
        batch_size,
        feature_dim,
    )

    # Simulated continuous hash codes
    hash_codes = torch.tanh(
        torch.randn(
            batch_size,
            hash_bits,
            requires_grad=True,
        )
    )

    # Simulated classification outputs
    class_logits = torch.randn(
        batch_size,
        num_classes,
        requires_grad=True,
    )

    # Example labels
    labels = torch.tensor(
        [0, 0, 1, 1, 2, 2, 3, 3]
    )

    # Construct pairwise similarity labels
    similarity_labels = (
        labels[:, None] == labels[None, :]
    ).float()

    # Calculate all losses
    total, losses = total_hashing_loss(
        hash_codes=hash_codes,
        similarity_labels=similarity_labels,
        class_logits=class_logits,
        labels=labels,
        student_features=student_features,
        projected_teacher=projected_teacher,
    )

    print("\n===== HASHING LOSS TEST =====")

    for name, value in losses.items():
        print(f"{name}: {value.item():.6f}")

    print(f"\nTotal loss: {total.item():.6f}")

    # Verify backward propagation
    total.backward()

    print("\nBackward propagation successful!")


if __name__ == "__main__":
    main()
