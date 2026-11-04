
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from models.student import StudentCNN
from models.hashing import HashingLayer
from models.hashing_losses import total_hashing_loss
from models.distillation import TeacherProjection


# ==========================================
# EXPERIMENT SETTINGS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data" / "cifar10"
CACHE_PATH = PROJECT_ROOT / "data" / "teacher_features_1000.pt"

KD_CHECKPOINT_PATH = PROJECT_ROOT / "distilled_student_v2.pth"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
MODEL_PATH = PROJECT_ROOT / "hashing_model_32bit_kdv2.pth"

NUM_IMAGES = 1000
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0005

HASH_BITS = 32
SEED = 42

ALPHA = 0.1
BETA = 1.0
LAMBDA_D = 0.1
GAMMA = 1.0
ETA = 1.0


# ==========================================
# DATASET WITH INDICES
# ==========================================

class IndexedDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label, index


# ==========================================
# MAIN
# ==========================================

def main():

    torch.manual_seed(SEED)

    device = torch.device("cpu")

    print("Using device:", device)

    # ======================================
    # LOAD CACHED TEACHER FEATURES
    # ======================================

    if not CACHE_PATH.exists():
        raise FileNotFoundError("Teacher feature cache not found.")

    cache = torch.load(
        CACHE_PATH,
        map_location="cpu",
        weights_only=True,
    )

    teacher_features = cache["features"].float()
    cached_labels = cache["labels"].long()

    print("Teacher features:", teacher_features.shape)

    # ======================================
    # LOAD KD-V2 CHECKPOINT
    # ======================================

    if not KD_CHECKPOINT_PATH.exists():
        raise FileNotFoundError("KD-v2 checkpoint not found.")

    kd_checkpoint = torch.load(
        KD_CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    # Load the previously trained student
    student = StudentCNN(feature_dim=256).to(device)

    student.load_state_dict(
        kd_checkpoint["student_state_dict"]
    )

    # Load the previously trained teacher projection
    projection = TeacherProjection(
        teacher_dim=512,
        student_dim=256,
    ).to(device)

    projection.projection.load_state_dict(
        kd_checkpoint["projection_state_dict"]
    )

    projection.eval()

    for parameter in projection.parameters():
        parameter.requires_grad = False

    # Project cached teacher features
    with torch.no_grad():
        projected_teacher = projection(teacher_features)

    print("KD-v2 student loaded successfully.")
    print("KD-v2 projection loaded successfully.")
    print("Projected teacher shape:", projected_teacher.shape)

    # ======================================
    # LOAD CIFAR-10
    # ======================================

    # Must match the preprocessing used during KD-v2
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transform,
    )

    current_labels = torch.tensor(
        dataset.targets[:NUM_IMAGES]
    )

    if not torch.equal(current_labels, cached_labels):
        raise ValueError("Cached labels do not match CIFAR-10.")

    indexed_dataset = IndexedDataset(
        torch.utils.data.Subset(
            dataset,
            range(NUM_IMAGES),
        )
    )

    loader = DataLoader(
        indexed_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    # ======================================
    # CREATE NEW HASHING AND CLASSIFICATION HEADS
    # ======================================

    hashing_layer = HashingLayer(
        input_dim=256,
        hash_bits=HASH_BITS,
    ).to(device)

    classifier = nn.Linear(256, 10).to(device)

    # Student starts from KD-v2 weights.
    # Student, hashing layer and classifier are trainable.
    optimizer = torch.optim.Adam(
        list(student.parameters())
        + list(hashing_layer.parameters())
        + list(classifier.parameters()),
        lr=LEARNING_RATE,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nStarting KD-v2 hashing training...", flush=True)
    print("Training images:", NUM_IMAGES)
    print("Hash bits:", HASH_BITS)
    print("Batches per epoch:", len(loader))

    # ======================================
    # TRAINING LOOP
    # ======================================

    for epoch in range(EPOCHS):

        student.train()
        hashing_layer.train()
        classifier.train()

        total_epoch_loss = 0.0

        loss_totals = {
            "similarity": 0.0,
            "quantization": 0.0,
            "balance": 0.0,
            "decorrelation": 0.0,
            "classification": 0.0,
            "distillation": 0.0,
        }

        correct = 0
        total = 0

        for batch_index, (images, labels, indices) in enumerate(
            loader, start=1
        ):

            images = images.to(device)
            labels = labels.to(device)
            indices = indices.long()

            teacher_target = projected_teacher[indices]

            # Student features
            student_features = student(images)

            # Continuous hash codes
            continuous_codes = hashing_layer(student_features)

            # Classification output
            class_logits = classifier(student_features)

            # Same-class images are treated as similar
            similarity_labels = (
                labels[:, None] == labels[None, :]
            ).float()

            # Calculate all six losses
            loss, individual_losses = total_hashing_loss(
                hash_codes=continuous_codes,
                similarity_labels=similarity_labels,
                class_logits=class_logits,
                labels=labels,
                student_features=student_features,
                projected_teacher=teacher_target,
                alpha=ALPHA,
                beta=BETA,
                lambda_d=LAMBDA_D,
                gamma=GAMMA,
                eta=ETA,
            )

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record losses
            total_epoch_loss += loss.item()

            for name, value in individual_losses.items():
                loss_totals[name] += value.item()

            predictions = class_logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            if batch_index % 10 == 0 or batch_index == len(loader):

                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] "
                    f"Batch [{batch_index}/{len(loader)}] "
                    f"Loss: {total_epoch_loss / batch_index:.4f}",
                    flush=True,
                )

        # ==================================
        # EPOCH SUMMARY
        # ==================================

        average_loss = total_epoch_loss / len(loader)
        accuracy = 100 * correct / total

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}] completed")
        print(f"Total loss: {average_loss:.6f}")

        for name, value in loss_totals.items():
            print(
                f"{name} loss: {value / len(loader):.6f}"
            )

        print(f"Training accuracy: {accuracy:.2f}%")

        # ==================================
        # SAVE CHECKPOINT AFTER EACH EPOCH
        # ==================================

        checkpoint = {
            "epoch": epoch + 1,
            "student_state_dict": student.state_dict(),
            "hashing_state_dict": hashing_layer.state_dict(),
            "classifier_state_dict": classifier.state_dict(),
            "projection_state_dict": projection.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "hash_bits": HASH_BITS,
            "seed": SEED,
        }

        checkpoint_path = (
            CHECKPOINT_DIR / f"hashing_kdv2_epoch_{epoch + 1}.pth"
        )

        torch.save(checkpoint, checkpoint_path)

        print("Checkpoint saved:", checkpoint_path)

    # ======================================
    # SAVE FINAL MODEL
    # ======================================

    torch.save(
        {
            "student_state_dict": student.state_dict(),
            "hashing_state_dict": hashing_layer.state_dict(),
            "classifier_state_dict": classifier.state_dict(),
            "projection_state_dict": projection.state_dict(),
            "hash_bits": HASH_BITS,
            "seed": SEED,
        },
        MODEL_PATH,
    )

    print("\nKD-v2 hashing training completed.")
    print("Final model saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()
