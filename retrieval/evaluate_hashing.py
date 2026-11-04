
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.student import StudentCNN
from models.hashing import HashingLayer


# =====================================================
# 1. EXPERIMENT SETTINGS
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "cifar10"

MODEL_PATH = ROOT / "hashing_model_32bit_kdv2.pth"

DATABASE_SIZE = 1000
QUERY_SIZE = 100

BATCH_SIZE = 32
TOP_K = 5

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# =====================================================
# 2. EXTRACT FEATURES AND HASH CODES
# =====================================================

def extract_outputs(
    student,
    hashing_layer,
    dataset,
    device,
):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    all_features = []
    all_continuous_codes = []
    all_binary_codes = []

    student.eval()
    hashing_layer.eval()

    print("Extracting student features and hash codes...")

    with torch.no_grad():

        for batch_index, (images, labels) in enumerate(
            loader,
            start=1,
        ):

            images = images.to(device)

            # Step 1: Extract student features
            features = student(images)

            # Step 2: Obtain continuous hash values
            continuous_codes = hashing_layer(features)

            # Step 3: Convert continuous values into binary codes
            binary_codes = torch.where(
                continuous_codes >= 0,
                torch.ones_like(continuous_codes),
                -torch.ones_like(continuous_codes),
            )

            all_features.append(features.cpu())
            all_continuous_codes.append(
                continuous_codes.cpu()
            )
            all_binary_codes.append(binary_codes.cpu())

            processed = min(
                batch_index * BATCH_SIZE,
                len(dataset),
            )

            print(
                f"Batch [{batch_index}/{len(loader)}] "
                f"Images processed: {processed}/{len(dataset)}",
                flush=True,
            )

    features = torch.cat(all_features, dim=0)

    continuous_codes = torch.cat(
        all_continuous_codes,
        dim=0,
    )

    binary_codes = torch.cat(
        all_binary_codes,
        dim=0,
    )

    return features, continuous_codes, binary_codes


# =====================================================
# 3. HAMMING DISTANCE
# =====================================================

def hamming_distance(query_codes, database_codes):

    distances = (
        query_codes[:, None, :]
        != database_codes[None, :, :]
    ).sum(dim=2)

    return distances


# =====================================================
# 4. STUDENT FEATURE DIAGNOSTICS
# =====================================================

def diagnose_student_features(features):

    print("\n===== STUDENT FEATURE DIAGNOSTICS =====")

    print("Feature shape:", features.shape)

    print(
        "Feature mean:",
        features.mean().item(),
    )

    print(
        "Feature standard deviation:",
        features.std().item(),
    )

    print(
        "Average feature-wise standard deviation:",
        features.std(dim=0).mean().item(),
    )

    # Normalize feature vectors for cosine similarity
    normalized = F.normalize(
        features,
        p=2,
        dim=1,
    )

    # Pairwise cosine similarities
    similarity_matrix = normalized @ normalized.T

    # Exclude self-similarity
    mask = ~torch.eye(
        len(features),
        dtype=torch.bool,
    )

    pairwise_similarities = similarity_matrix[mask]

    print(
        "Average pairwise feature cosine similarity:",
        pairwise_similarities.mean().item(),
    )

    print(
        "Minimum pairwise feature cosine similarity:",
        pairwise_similarities.min().item(),
    )

    print(
        "Maximum pairwise feature cosine similarity:",
        pairwise_similarities.max().item(),
    )


# =====================================================
# 5. CONTINUOUS HASH DIAGNOSTICS
# =====================================================

def diagnose_continuous_codes(continuous_codes):

    print("\n===== CONTINUOUS HASH DIAGNOSTICS =====")

    print(
        "Continuous hash shape:",
        continuous_codes.shape,
    )

    print(
        "Continuous hash mean:",
        continuous_codes.mean().item(),
    )

    print(
        "Continuous hash standard deviation:",
        continuous_codes.std().item(),
    )

    print(
        "Average per-bit standard deviation:",
        continuous_codes.std(dim=0).mean().item(),
    )

    print(
        "Minimum continuous hash value:",
        continuous_codes.min().item(),
    )

    print(
        "Maximum continuous hash value:",
        continuous_codes.max().item(),
    )

    print(
        "Average absolute continuous hash value:",
        continuous_codes.abs().mean().item(),
    )

    # Positive proportion before binary conversion
    positive_ratios = (
        continuous_codes >= 0
    ).float().mean(dim=0)

    print("\nPositive ratio for each continuous hash bit:")

    for bit_index, ratio in enumerate(positive_ratios):

        print(
            f"Bit {bit_index + 1:02d}: {ratio.item():.4f}"
        )


# =====================================================
# 6. BINARY HASH CODE DIAGNOSTICS
# =====================================================

def diagnose_binary_codes(database_codes):

    print("\n===== BINARY HASH CODE DIAGNOSTICS =====")

    # Count distinct binary codes
    unique_codes = torch.unique(
        database_codes,
        dim=0,
    )

    print(
        "Unique database codes:",
        len(unique_codes),
    )

    print(
        "Total database images:",
        len(database_codes),
    )

    print(
        "Percentage of unique codes:",
        f"{100 * len(unique_codes) / len(database_codes):.2f}%",
    )

    # Check individual bits
    positive_ratios = (
        database_codes == 1
    ).float().mean(dim=0)

    print(
        "\nAverage positive-bit ratio:",
        positive_ratios.mean().item(),
    )

    print(
        "Minimum positive-bit ratio:",
        positive_ratios.min().item(),
    )

    print(
        "Maximum positive-bit ratio:",
        positive_ratios.max().item(),
    )

    # Pairwise Hamming distance
    print("\nCalculating database pairwise distances...")

    distances = hamming_distance(
        database_codes,
        database_codes,
    )

    # Remove comparisons of an image with itself
    mask = ~torch.eye(
        len(database_codes),
        dtype=torch.bool,
    )

    pairwise_distances = distances[mask].float()

    print(
        "\nAverage pairwise Hamming distance:",
        pairwise_distances.mean().item(),
    )

    print(
        "Minimum pairwise Hamming distance:",
        pairwise_distances.min().item(),
    )

    print(
        "Maximum pairwise Hamming distance:",
        pairwise_distances.max().item(),
    )

    print(
        "Hamming distance standard deviation:",
        pairwise_distances.std().item(),
    )

    # Percentage of identical codes among different images
    zero_pairs = (
        pairwise_distances == 0
    ).sum().item()

    total_pairs = len(pairwise_distances)

    print(
        "\nZero-distance pair percentage:",
        f"{100 * zero_pairs / total_pairs:.2f}%",
    )


# =====================================================
# 7. RETRIEVAL METRICS
# =====================================================

def calculate_metrics(
    distances,
    query_labels,
    database_labels,
):

    average_precisions = []
    precision_at_5 = []
    precision_at_10 = []

    for i in range(len(query_labels)):

        query_label = query_labels[i]

        # Sort by ascending Hamming distance
        ranking = torch.argsort(distances[i])

        retrieved_labels = database_labels[ranking]

        # Same-class images are relevant
        relevant = (
            retrieved_labels == query_label
        ).float()

        total_relevant = relevant.sum().item()

        if total_relevant == 0:
            continue

        cumulative_relevant = torch.cumsum(
            relevant,
            dim=0,
        )

        positions = torch.arange(
            1,
            len(relevant) + 1,
            dtype=torch.float32,
        )

        precision_at_positions = (
            cumulative_relevant / positions
        )

        average_precision = (
            precision_at_positions * relevant
        ).sum() / total_relevant

        average_precisions.append(
            average_precision.item()
        )

        precision_at_5.append(
            relevant[:5].mean().item()
        )

        precision_at_10.append(
            relevant[:10].mean().item()
        )

    if not average_precisions:
        raise ValueError(
            "No query has a relevant database image."
        )

    map_score = (
        sum(average_precisions)
        / len(average_precisions)
    )

    p5 = (
        sum(precision_at_5)
        / len(precision_at_5)
    )

    p10 = (
        sum(precision_at_10)
        / len(precision_at_10)
    )

    return map_score, p5, p10


# =====================================================
# 8. DISPLAY RETRIEVAL RESULTS
# =====================================================

def show_retrieval(
    query_dataset,
    database_dataset,
    query_index,
    distances,
):

    ranking = torch.argsort(
        distances[query_index]
    )

    retrieved_indices = ranking[:TOP_K]

    query_image = query_dataset.data[query_index]

    query_label = query_dataset.targets[query_index]

    fig, axes = plt.subplots(
        1,
        TOP_K + 1,
        figsize=(16, 4),
    )

    # Query image
    axes[0].imshow(query_image)

    axes[0].set_title(
        f"QUERY\n{CLASS_NAMES[query_label]}"
    )

    axes[0].axis("off")

    # Retrieved images
    for rank, db_index in enumerate(
        retrieved_indices,
        start=1,
    ):

        db_index = db_index.item()

        image = database_dataset.data[db_index]

        label = database_dataset.targets[db_index]

        distance = distances[
            query_index,
            db_index,
        ].item()

        axes[rank].imshow(image)

        axes[rank].set_title(
            f"Rank {rank}\n"
            f"{CLASS_NAMES[label]}\n"
            f"Hamming: {distance}"
        )

        axes[rank].axis("off")

    plt.tight_layout()
    plt.show()


# =====================================================
# 9. MAIN
# =====================================================

def main():

    device = torch.device("cpu")

    print("Using device:", device)

    # ---------------------------------------------
    # Load trained checkpoint
    # ---------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    hash_bits = checkpoint["hash_bits"]

    student = StudentCNN(
        feature_dim=256
    ).to(device)

    hashing_layer = HashingLayer(
        input_dim=256,
        hash_bits=hash_bits,
    ).to(device)

    student.load_state_dict(
        checkpoint["student_state_dict"]
    )

    hashing_layer.load_state_dict(
        checkpoint["hashing_state_dict"]
    )

    print("Trained student loaded successfully.")
    print("Trained hashing layer loaded successfully.")
    print("Hash bits:", hash_bits)

    # ---------------------------------------------
    # Preprocessing
    # ---------------------------------------------

    transform = transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),

    ])

    # ---------------------------------------------
    # Load CIFAR-10
    # ---------------------------------------------

    database_full = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transform,
    )

    query_full = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transform,
    )

    database_dataset = Subset(
        database_full,
        range(DATABASE_SIZE),
    )

    query_dataset = Subset(
        query_full,
        range(QUERY_SIZE),
    )

    # ---------------------------------------------
    # Extract database outputs
    # ---------------------------------------------

    print("\nExtracting database features...")

    (
        database_features,
        database_continuous,
        database_codes,
    ) = extract_outputs(
        student,
        hashing_layer,
        database_dataset,
        device,
    )

    # ---------------------------------------------
    # Extract query outputs
    # ---------------------------------------------

    print("\nExtracting query features...")

    (
        query_features,
        query_continuous,
        query_codes,
    ) = extract_outputs(
        student,
        hashing_layer,
        query_dataset,
        device,
    )

    database_labels = torch.tensor(
        database_full.targets[:DATABASE_SIZE]
    )

    query_labels = torch.tensor(
        query_full.targets[:QUERY_SIZE]
    )

    print("\nDatabase feature shape:", database_features.shape)
    print("Query feature shape:", query_features.shape)

    print(
        "Database continuous hash shape:",
        database_continuous.shape,
    )

    print(
        "Database binary code shape:",
        database_codes.shape,
    )

    print(
        "Query binary code shape:",
        query_codes.shape,
    )

    print(
        "Unique binary values:",
        torch.unique(database_codes),
    )

    # ---------------------------------------------
    # Run all diagnostics
    # ---------------------------------------------

    diagnose_student_features(database_features)

    diagnose_continuous_codes(database_continuous)

    diagnose_binary_codes(database_codes)

    # ---------------------------------------------
    # Hamming distance
    # ---------------------------------------------

    print("\nCalculating query-database Hamming distances...")

    distances = hamming_distance(
        query_codes,
        database_codes,
    )

    # ---------------------------------------------
    # Evaluate retrieval
    # ---------------------------------------------

    map_score, p5, p10 = calculate_metrics(
        distances,
        query_labels,
        database_labels,
    )

    print("\n===== 32-BIT BINARY HASHING EVALUATION =====")

    print("Database images:", DATABASE_SIZE)
    print("Query images:", QUERY_SIZE)
    print("Hash bits:", hash_bits)

    print(f"mAP: {map_score:.4f}")
    print(f"Precision@5: {p5:.4f}")
    print(f"Precision@10: {p10:.4f}")

    # ---------------------------------------------
    # Show one retrieval example
    # ---------------------------------------------

    query_index = 0

    print(
        "\nQuery class:",
        CLASS_NAMES[query_labels[query_index].item()],
    )

    show_retrieval(
        query_full,
        database_full,
        query_index,
        distances,
    )


if __name__ == "__main__":
    main()
