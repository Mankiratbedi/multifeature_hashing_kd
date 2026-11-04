
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.teacher import ResNet18Teacher
from models.student import StudentCNN
from models.distillation import TeacherProjection


# Experiment settings
DATA_DIR = "data/cifar10"
NUM_TRAIN_IMAGES = 1000
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001


# Preprocessing for pretrained ResNet-18
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


def main():
    device = torch.device("cpu")
    print("Using device:", device)

    # Load CIFAR-10 training data
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

    # Frozen teacher
    teacher = ResNet18Teacher().to(device)
    teacher.eval()

    # Trainable student and teacher projection
    student = StudentCNN(feature_dim=256).to(device)

    teacher_projection = TeacherProjection(
        teacher_dim=512,
        student_dim=256,
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        list(student.parameters())
        + list(teacher_projection.parameters()),
        lr=LEARNING_RATE,
    )

    print("\nStarting knowledge distillation...", flush=True)
    print(f"Training images: {len(subset)}", flush=True)
    print(f"Batches per epoch: {len(loader)}", flush=True)

    # Train for the specified number of epochs
    for epoch in range(EPOCHS):
        student.train()
        teacher_projection.train()

        total_loss = 0.0

        for batch_index, (images, _) in enumerate(loader, start=1):
            images = images.to(device)

            # Extract teacher features without updating teacher weights
            with torch.no_grad():
                teacher_features = teacher(images)

            # Project teacher features to 256 dimensions
            projected_teacher = teacher_projection(teacher_features)

            # Extract student features
            student_features = student(images)

            # Calculate distillation loss
            loss = criterion(
                student_features,
                projected_teacher,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Display progress every 10 batches and on the last batch
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

    # Save the trained student and projection after all epochs
    torch.save(
        {
            "student_state_dict": student.state_dict(),
            "projection_state_dict": teacher_projection.state_dict(),
        },
        "distilled_student.pth",
    )

    print("\nTraining finished.")
    print("Saved model to distilled_student.pth")


if __name__ == "__main__":
    main()
