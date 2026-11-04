
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.teacher import ResNet18Teacher


# -------------------------------
# 1. Configuration
# -------------------------------
DATABASE_SIZE = 1000
QUERY_SIZE = 100
BATCH_SIZE = 32
TOP_K = 5

DATA_DIR = "data/cifar10"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Use ImageNet normalization for the pretrained ResNet-18.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


# -------------------------------
# 2. Extract image features
# -------------------------------
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
    start_time = time.time()

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            features = model(images)

            # Normalize vectors for cosine similarity.
            features = F.normalize(features, p=2, dim=1)

            all_features.append(features.cpu())

    features = torch.cat(all_features, dim=0)

    elapsed = time.time() - start_time
    print(
        f"Extracted {len(indices)} images "
        f"in {elapsed:.2f} seconds."
    )

    return features


# -------------------------------
# 3. Retrieve similar images
# -------------------------------
def retrieve(query_features, database_features, query_index):
    query_vector = query_features[query_index]

    # Each score is the cosine similarity to one database image.
    scores = database_features @ query_vector

    top_scores, top_indices = torch.topk(
        scores,
        k=TOP_K,
    )

    return top_indices.numpy(), top_scores.numpy()


# -------------------------------
# 4. Visualize retrieval results
# -------------------------------
def show_results(
    query_dataset,
    database_dataset,
    query_index,
    retrieved_indices,
    scores,
):
    query_image = query_dataset.data[query_index]
    query_label = query_dataset.targets[query_index]

    fig, axes = plt.subplots(1, TOP_K + 1, figsize=(15, 4))

    axes[0].imshow(query_image)
    axes[0].set_title(
        f"QUERY\n{CLASS_NAMES[query_label]}"
    )
    axes[0].axis("off")

    for rank, (db_index, score) in enumerate(
        zip(retrieved_indices, scores),
        start=1,
    ):
        image = database_dataset.data[int(db_index)]
        label = database_dataset.targets[int(db_index)]

        axes[rank].imshow(image)
        axes[rank].set_title(
            f"Rank {rank}\n"
            f"{CLASS_NAMES[label]}\n"
            f"Similarity: {score:.3f}"
        )
        axes[rank].axis("off")

    plt.tight_layout()
    plt.show()


# -------------------------------
# 5. Run the baseline experiment
# -------------------------------
def main():
    device = torch.device("cpu")
    print("Using device:", device)

    # Separate training database and test queries.
    database_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )

    query_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )

    database_indices = list(range(DATABASE_SIZE))
    query_indices = list(range(QUERY_SIZE))

    # Load pretrained ResNet-18.
    model = ResNet18Teacher().to(device)
    model.eval()

    print("\nExtracting database features...")
    database_features = extract_features(
        model,
        database_dataset,
        database_indices,
        device,
    )

    print("\nExtracting query features...")
    query_features = extract_features(
        model,
        query_dataset,
        query_indices,
        device,
    )

    print("\nDatabase feature shape:", database_features.shape)
    print("Query feature shape:", query_features.shape)

    # Use the first test image as an example query.
    query_index = 0

    retrieved_indices, scores = retrieve(
        query_features,
        database_features,
        query_index,
    )

    query_label = query_dataset.targets[query_index]

    print("\nQuery class:", CLASS_NAMES[query_label])
    print("Top retrieved results:")

    for rank, (index, score) in enumerate(
        zip(retrieved_indices, scores),
        start=1,
    ):
        label = database_dataset.targets[int(index)]

        print(
            f"{rank}. {CLASS_NAMES[label]} "
            f"(cosine similarity: {score:.4f})"
        )

    show_results(
        query_dataset,
        database_dataset,
        query_index,
        retrieved_indices,
        scores,
    )


if __name__ == "__main__":
    main()
