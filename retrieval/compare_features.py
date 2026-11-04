
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.teacher import ResNet18Teacher
from models.student import StudentCNN


DATA_DIR = "data/cifar10"
NUM_IMAGES = 1000
BATCH_SIZE = 32


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


def extract_features(model, loader):
    features = []

    model.eval()

    with torch.no_grad():
        for images, _ in loader:
            output = model(images)
            features.append(output)

    return torch.cat(features, dim=0)


def calculate_similarity(features):
    normalized = F.normalize(features, p=2, dim=1)

    similarity = normalized @ normalized.T

    mask = ~torch.eye(
        len(features),
        dtype=torch.bool,
    )

    return similarity[mask].mean().item()


def main():
    device = torch.device("cpu")

    dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )

    loader = DataLoader(
        Subset(dataset, range(NUM_IMAGES)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Load teacher
    teacher = ResNet18Teacher().to(device)

    # Load trained student
    student = StudentCNN(feature_dim=256).to(device)

    checkpoint = torch.load(
        "distilled_student.pth",
        map_location=device,
        weights_only=True,
    )

    student.load_state_dict(
        checkpoint["student_state_dict"]
    )

    print("Extracting teacher features...")
    teacher_features = extract_features(teacher, loader)

    print("Extracting student features...")
    student_features = extract_features(student, loader)

    teacher_similarity = calculate_similarity(
        teacher_features
    )

    student_similarity = calculate_similarity(
        student_features
    )

    print("\n===== FEATURE COMPARISON =====")

    print(
        "Teacher average pairwise similarity:",
        round(teacher_similarity, 6),
    )

    print(
        "Student average pairwise similarity:",
        round(student_similarity, 6),
    )


if __name__ == "__main__":
    main()

