
import torch
from retrieval.baseline import (
    DATABASE_SIZE,
    QUERY_SIZE,
    BATCH_SIZE,
    DATA_DIR,
    CLASS_NAMES,
    extract_features,
)
from models.teacher import ResNet18Teacher
from torchvision import datasets, transforms


def average_precision(relevant):
    """Calculate AP from a ranked list of relevant/not-relevant results."""
    relevant = torch.as_tensor(relevant, dtype=torch.float32)

    total_relevant = relevant.sum().item()
    if total_relevant == 0:
        return 0.0

    cumulative_relevant = torch.cumsum(relevant, dim=0)
    ranks = torch.arange(1, len(relevant) + 1, dtype=torch.float32)

    precision_at_ranks = cumulative_relevant / ranks
    ap = (precision_at_ranks * relevant).sum() / total_relevant

    return ap.item()


def main():
    device = torch.device("cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    database = datasets.CIFAR10(
        root=DATA_DIR, train=True,
        download=True, transform=transform
    )
    queries = datasets.CIFAR10(
        root=DATA_DIR, train=False,
        download=True, transform=transform
    )

    database_indices = list(range(DATABASE_SIZE))
    query_indices = list(range(QUERY_SIZE))

    model = ResNet18Teacher().to(device)
    model.eval()

    print("\nExtracting database features...")
    db_features = extract_features(
        model, database, database_indices, device
    )

    print("\nExtracting query features...")
    query_features = extract_features(
        model, queries, query_indices, device
    )

    # Labels for the selected images
    db_labels = torch.tensor(
        [database.targets[i] for i in database_indices]
    )
    query_labels = torch.tensor(
        [queries.targets[i] for i in query_indices]
    )

    # Cosine similarity: features are already L2-normalized
    similarities = query_features @ db_features.T

    average_precisions = []
    precisions_at_5 = []

    for i in range(len(query_indices)):
        # Sort database images from most to least similar
        ranking = torch.argsort(similarities[i], descending=True)

        # An image is relevant if it has the same class label
        relevant = (
            db_labels[ranking] == query_labels[i]
        ).float()

        ap = average_precision(relevant)
        average_precisions.append(ap)

        precision_at_5 = relevant[:5].mean().item()
        precisions_at_5.append(precision_at_5)

    map_score = sum(average_precisions) / len(average_precisions)
    mean_precision_at_5 = (
        sum(precisions_at_5) / len(precisions_at_5)
    )

    print("\n===== BASELINE EVALUATION =====")
    print(f"Database images: {DATABASE_SIZE}")
    print(f"Query images: {QUERY_SIZE}")
    print(f"Feature dimension: {db_features.shape[1]}")
    print(f"Mean Precision@5: {mean_precision_at_5:.4f}")
    print(f"mAP: {map_score:.4f}")


if __name__ == "__main__":
    main()
