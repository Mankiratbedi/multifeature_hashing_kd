
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.teacher import ResNet18Teacher
from models.student import StudentCNN


# ==========================================
# EXPERIMENT SETTINGS
# ==========================================

DATA_DIR = "data/cifar10"
NUM_TRAIN_IMAGES = 1000
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
LAMBDA_CLS = 1.0
SEED = 42

MODEL_PATH = "distilled_student_v3.pth"


def main():

    torch.manual_seed(SEED)

    device = torch.device("cpu")
    print("Using device:", device)

    # ======================================
    # DATASET AND PREPROCESSING
    # ======================================

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )

    subset = Subset(dataset, range(NUM_TRAIN_IMAGES))

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    # ======================================
    # FROZEN TEACHER
    # ======================================

    teacher = ResNet18Teacher().to(device)
    teacher.eval()

    # Fixed teacher projection
    projection = nn.Linear(512, 256).to(device)

    projection.eval()

    for parameter in projection.parameters():
        parameter.requires_grad = False

    # ======================================
    # STUDENT AND CLASSIFIER
    # ======================================

    student = StudentCNN(feature_dim=256).to(device)

    # Classification head: 256 features -> 10 classes
    classifier = nn.Linear(256, 10).to(device)

    # ======================================
    # LOSS FUNCTIONS AND OPTIMIZER
    # ======================================

    classification_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        list(student.parameters())
        + list(classifier.parameters()),
        lr=LEARNING_RATE,
    )

    print("\nStarting KD-v3 training...", flush=True)
    print("Training images:", len(subset), flush=True)
    print("Batches per epoch:", len(loader), flush=True)

    # ======================================
    # TRAINING LOOP
    # ======================================

    for epoch in range(EPOCHS):

        student.train()
        classifier.train()

        total_loss = 0.0
        total_distill_loss = 0.0
        total_classification_loss = 0.0
        correct = 0
        total = 0

        for batch_index, (images, labels) in enumerate(
            loader, start=1
        ):

            images = images.to(device)
            labels = labels.to(device)

            # Teacher features (no gradient updates)
            with torch.no_grad():

                teacher_features = teacher(images)

                projected_teacher = projection(
                    teacher_features
                )

                projected_teacher = F.normalize(
                    projected_teacher,
                    p=2,
                    dim=1,
                )

            # Student features
            student_features = student(images)

            normalized_student = F.normalize(
                student_features,
                p=2,
                dim=1,
            )

            # Distillation loss: cosine distance
            distill_loss = (
                1 - F.cosine_similarity(
                    normalized_student,
                    projected_teacher,
                    dim=1,
                )
            ).mean()

            # Classification loss
            logits = classifier(student_features)

            classification_loss = classification_criterion(
                logits,
                labels,
            )

            # Combined objective
            loss = (
                distill_loss
                + LAMBDA_CLS * classification_loss
            )

            # Update student and classifier
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # Track statistics
            total_loss += loss.item()
            total_distill_loss += distill_loss.item()
            total_classification_loss += classification_loss.item()

            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Progress display
            if batch_index % 10 == 0 or batch_index == len(loader):

                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] "
                    f"Batch [{batch_index}/{len(loader)}] "
                    f"Total Loss: {total_loss / batch_index:.4f} "
                    f"Distill: {total_distill_loss / batch_index:.4f} "
                    f"Class: {total_classification_loss / batch_index:.4f}",
                    flush=True,
                )

        # Epoch statistics
        average_loss = total_loss / len(loader)
        average_distill = total_distill_loss / len(loader)
        average_classification = (
            total_classification_loss / len(loader)
        )

        accuracy = 100 * correct / total

        print(
            f"\nEpoch [{epoch + 1}/{EPOCHS}] completed"
        )

        print(f"Average total loss: {average_loss:.6f}")
        print(f"Average distillation loss: {average_distill:.6f}")
        print(f"Average classification loss: {average_classification:.6f}")
        print(f"Training accuracy: {accuracy:.2f}%\n")

    # ======================================
    # SAVE TRAINED MODEL
    # ======================================

    torch.save(
        {
            "student_state_dict": student.state_dict(),
            "classifier_state_dict": classifier.state_dict(),
            "projection_state_dict": projection.state_dict(),
            "seed": SEED,
            "lambda_cls": LAMBDA_CLS,
        },
        MODEL_PATH,
    )

    print("KD-v3 training finished.")
    print("Saved model to", MODEL_PATH)


if __name__ == "__main__":
    main()
