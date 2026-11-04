
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.student import StudentCNN


# Experiment settings
DATA_DIR = "data/cifar10"
MODEL_PATH = "distilled_student.pth"

DATABASE_SIZE = 1000
QUERY_SIZE = 100
BATCH_SIZE = 32
TOP_K = 5

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# Same preprocessing used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


def extract_features(model, dataset, indices, device):

    subset = Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    all_features = []

    model.eval()

    with torch.no_grad():
        for images, _ in loader:

            images = images.to(device)

            features = model(images)

            # Normalize feature vectors for cosine similarity
            features = F.normalize(features, p=2, dim=1)

            all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)


def average_precision(relevant):

    relevant = relevant.float()

    total_relevant = relevant.sum().item()

    if total_relevant == 0:
        return 0.0

    cumulative_relevant = torch.cumsum(relevant, dim=0)

    ranks = torch.arange(
        1,
        len(relevant) + 1,
        dtype=torch.float32
    )

    precision_at_ranks = cumulative_relevant / ranks

    ap = (precision_at_ranks * relevant).sum() / total_relevant

    return ap.item()


def main():

    device = torch.device("cpu")

    print("Using device:", device)

    # Load CIFAR-10
    database = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )

    queries = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )

    database_indices = list(range(DATABASE_SIZE))
    query_indices = list(range(QUERY_SIZE))

    # Create student model
    student = StudentCNN(feature_dim=256).to(device)

    # Load trained student weights
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    student.load_state_dict(checkpoint["student_state_dict"])
    student.eval()

    print("\nStudent model loaded successfully.")

    # Extract database features
    print("\nExtracting database features...")

    database_features = extract_features(
        student,
        database,
        database_indices,
        device,
    )

    # Extract query features
    print("\nExtracting query features...")

    query_features = extract_features(
        student,
        queries,
        query_indices,
        device,
    )

    print("\nDatabase feature shape:", database_features.shape)
    print("Query feature shape:", query_features.shape)

    # Get labels
    database_labels = torch.tensor(
        [database.targets[i] for i in database_indices]
    )

    query_labels = torch.tensor(
        [queries.targets[i] for i in query_indices]
    )

    # Cosine similarity
    similarities = query_features @ database_features.T

    average_precisions = []
    precisions_at_5 = []

    # Evaluate all queries
    for i in range(QUERY_SIZE):

        ranking = torch.argsort(
            similarities[i],
            descending=True
        )

        relevant = (
            database_labels[ranking] == query_labels[i]
        ).float()

        ap = average_precision(relevant)

        average_precisions.append(ap)

        precision_at_5 = relevant[:TOP_K].mean().item()

        precisions_at_5.append(precision_at_5)

    # Calculate final metrics
    mean_ap = sum(average_precisions) / len(average_precisions)

    mean_precision = sum(precisions_at_5) / len(precisions_at_5)

    print("\n===== DISTILLED STUDENT EVALUATION =====")

    print("Database images:", DATABASE_SIZE)
    print("Query images:", QUERY_SIZE)
    print("Feature dimension:", database_features.shape[1])

    print(f"Mean Precision@5: {mean_precision:.4f}")
    print(f"mAP: {mean_ap:.4f}")

    # Show one retrieval example
    query_index = 0

    ranking = torch.argsort(
        similarities[query_index],
        descending=True
    )

    print("\nQuery class:", CLASS_NAMES[query_labels[query_index]])

    print("Top 5 retrieved results:")

    for rank, index in enumerate(ranking[:TOP_K], start=1):

        label = database_labels[index].item()

        score = similarities[query_index, index].item()

        print(
            f"{rank}. {CLASS_NAMES[label]} "
            f"(cosine similarity: {score:.4f})"
        )


if __name__ == "__main__":
    main()
