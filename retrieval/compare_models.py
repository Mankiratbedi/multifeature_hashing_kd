
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.teacher import ResNet18Teacher
from models.student import StudentCNN
from models.hashing import HashingLayer


# ==========================================
# CONFIGURATION
# ==========================================

DEVICE = torch.device("cpu")

DATA_DIR = "data/cifar10"

KD_CHECKPOINT = "distilled_student_v2.pth"
HASH_CHECKPOINT = "hashing_model_32bit_kdv2.pth"

DATABASE_SIZE = 1000
QUERY_SIZE = 100
BATCH_SIZE = 32
HASH_BITS = 32

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])


# ==========================================
# DATA LOADING
# ==========================================

def load_data():

    database = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=TRANSFORM
    )

    queries = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=TRANSFORM
    )

    db_indices = list(range(DATABASE_SIZE))
    query_indices = list(range(QUERY_SIZE))

    db_loader = DataLoader(
        Subset(database, db_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    query_loader = DataLoader(
        Subset(queries, query_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    db_labels = torch.tensor(
        [database.targets[i] for i in db_indices]
    )

    query_labels = torch.tensor(
        [queries.targets[i] for i in query_indices]
    )

    return db_loader, query_loader, db_labels, query_labels


# ==========================================
# LOAD MODELS
# ==========================================

def load_models():

    # Teacher
    teacher = ResNet18Teacher().to(DEVICE)
    teacher.eval()

    # KD-v2 student
    student = StudentCNN(feature_dim=256).to(DEVICE)

    kd_checkpoint = torch.load(
        KD_CHECKPOINT,
        map_location=DEVICE,
        weights_only=True
    )

    student.load_state_dict(
        kd_checkpoint["student_state_dict"]
    )
    student.eval()

    # Hashing model
    hashing_checkpoint = torch.load(
        HASH_CHECKPOINT,
        map_location=DEVICE,
        weights_only=True
    )

    hash_student = StudentCNN(feature_dim=256).to(DEVICE)
    hash_student.load_state_dict(
        hashing_checkpoint["student_state_dict"]
    )
    hash_student.eval()

    hashing_layer = HashingLayer(
        input_dim=256,
        hash_bits=HASH_BITS
    ).to(DEVICE)

    hashing_layer.load_state_dict(
        hashing_checkpoint["hashing_state_dict"]
    )
    hashing_layer.eval()

    return teacher, student, hash_student, hashing_layer


# ==========================================
# FEATURE / HASH EXTRACTION
# ==========================================

def extract_representations(
    model,
    loader,
    hashing_layer=None
):

    outputs = []
    start = time.time()

    with torch.no_grad():

        for images, _ in loader:

            images = images.to(DEVICE)

            features = model(images)

            if hashing_layer is None:
                representations = F.normalize(
                    features, p=2, dim=1
                )
            else:
                representations = hashing_layer(features)

                representations = torch.where(
                    representations >= 0,
                    1.0,
                    -1.0
                )

            outputs.append(representations.cpu())

    elapsed = time.time() - start

    return torch.cat(outputs), elapsed


# ==========================================
# RETRIEVAL METRICS
# ==========================================

def calculate_metrics(
    database_repr,
    query_repr,
    database_labels,
    query_labels,
    binary=False
):

    all_ap = []
    all_p5 = []
    all_p10 = []

    for i in range(len(query_repr)):

        if binary:

            distances = (
                database_repr != query_repr[i]
            ).sum(dim=1)

            ranking = torch.argsort(distances)

        else:

            similarities = (
                database_repr @ query_repr[i]
            )

            ranking = torch.argsort(
                similarities,
                descending=True
            )

        relevant = (
            database_labels[ranking] == query_labels[i]
        ).float()

        total_relevant = relevant.sum().item()

        if total_relevant == 0:
            continue

        cumulative = torch.cumsum(relevant, dim=0)

        ranks = torch.arange(
            1,
            len(relevant) + 1,
            dtype=torch.float32
        )

        precision = cumulative / ranks

        ap = (precision * relevant).sum() / total_relevant

        all_ap.append(ap.item())
        all_p5.append(relevant[:5].mean().item())
        all_p10.append(relevant[:10].mean().item())

    return (
        sum(all_ap) / len(all_ap),
        sum(all_p5) / len(all_p5),
        sum(all_p10) / len(all_p10)
    )


# ==========================================
# EVALUATE ONE MODEL
# ==========================================

def evaluate_model(
    name,
    model,
    db_loader,
    query_loader,
    db_labels,
    query_labels,
    hashing_layer=None
):

    print(f"\nEvaluating {name}...")

    db_repr, db_time = extract_representations(
        model,
        db_loader,
        hashing_layer
    )

    query_repr, query_time = extract_representations(
        model,
        query_loader,
        hashing_layer
    )

    binary = hashing_layer is not None

    metrics = calculate_metrics(
        db_repr,
        query_repr,
        db_labels,
        query_labels,
        binary
    )

    map_score, p5, p10 = metrics

    print("Representation shape:", db_repr.shape)
    print(f"mAP: {map_score:.4f}")
    print(f"Precision@5: {p5:.4f}")
    print(f"Precision@10: {p10:.4f}")
    print(f"Extraction time: {db_time + query_time:.2f} seconds")

    return {
        "name": name,
        "mAP": map_score,
        "P@5": p5,
        "P@10": p10,
        "dimension": db_repr.shape[1],
        "binary": binary,
        "time": db_time + query_time
    }


# ==========================================
# MAIN
# ==========================================

def main():

    print("Using device:", DEVICE)

    db_loader, query_loader, db_labels, query_labels = load_data()

    teacher, student, hash_student, hashing_layer = load_models()

    results = []

    results.append(
        evaluate_model(
            "ResNet-18 Teacher",
            teacher,
            db_loader,
            query_loader,
            db_labels,
            query_labels
        )
    )

    results.append(
        evaluate_model(
            "KD-v2 Student",
            student,
            db_loader,
            query_loader,
            db_labels,
            query_labels
        )
    )

    results.append(
        evaluate_model(
            "32-bit Hashing",
            hash_student,
            db_loader,
            query_loader,
            db_labels,
            query_labels,
            hashing_layer
        )
    )

    # ======================================
    # FINAL COMPARISON
    # ======================================

    print("\n" + "=" * 75)
    print("FINAL MODEL COMPARISON")
    print("=" * 75)

    print(
        f"{'Model':<22}"
        f"{'mAP':>10}"
        f"{'P@5':>10}"
        f"{'P@10':>10}"
        f"{'Dim':>8}"
        f"{'Time(s)':>12}"
    )

    for result in results:

        print(
            f"{result['name']:<22}"
            f"{result['mAP']:>10.4f}"
            f"{result['P@5']:>10.4f}"
            f"{result['P@10']:>10.4f}"
            f"{result['dimension']:>8}"
            f"{result['time']:>12.2f}"
        )


if __name__ == "__main__":
    main()
