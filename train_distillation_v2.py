
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.teacher import ResNet18Teacher
from models.student import StudentCNN


# Experiment settings
DATA_DIR = "data/cifar10"
NUM_TRAIN_IMAGES = 1000
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
SEED = 42

MODEL_PATH = "distilled_student_v2.pth"


def main():
    torch.manual_seed(SEED)

    device = torch.device("cpu")
    print("Using device:", device)

    # Preprocessing compatible with pretrained ResNet-18
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    # Load CIFAR-10
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

    # Load frozen teacher
    teacher = ResNet18Teacher().to(device)
    teacher.eval()

    # Create fixed teacher projection
    projection = nn.Linear(512, 256).to(device)

    # Freeze projection parameters
    projection.eval()
    for parameter in projection.parameters():
        parameter.requires_grad = False

    # Create trainable student
    student = StudentCNN(feature_dim=256).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=LEARNING_RATE,
    )

    print("\nStarting KD-v2 training...", flush=True)
    print("Training images:", len(subset), flush=True)
    print("Batches per epoch:", len(loader), flush=True)

    for epoch in range(EPOCHS):

        student.train()
        total_loss = 0.0

        for batch_index, (images, _) in enumerate(
            loader, start=1
        ):

            images = images.to(device)

            # Teacher and projection remain fixed
            with torch.no_grad():
                teacher_features = teacher(images)
                projected_teacher = projection(teacher_features)

                projected_teacher = F.normalize(
                    projected_teacher,
                    p=2,
                    dim=1,
                )

            # Student representation
            student_features = student(images)

            student_features = F.normalize(
                student_features,
                p=2,
                dim=1,
            )

            # Normalized feature matching loss
            loss = criterion(
                student_features,
                projected_teacher,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_index % 10 == 0 or batch_index == len(loader):
                running_loss = total_loss / batch_index

                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] "
                    f"Batch [{batch_index}/{len(loader)}] "
                    f"Running Loss: {running_loss:.6f}",
                    flush=True,
                )

        epoch_loss = total_loss / len(loader)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] completed. "
            f"Average Loss: {epoch_loss:.6f}",
            flush=True,
        )

    # Save student and fixed projection together
    torch.save(
        {
            "student_state_dict": student.state_dict(),
            "projection_state_dict": projection.state_dict(),
            "seed": SEED,
        },
        MODEL_PATH,
    )

    print("\nKD-v2 training finished.")
    print("Saved model to", MODEL_PATH)


if __name__ == "__main__":
    main()
