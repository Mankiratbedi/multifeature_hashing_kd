
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.student import StudentCNN


MODEL_PATH = "distilled_student.pth"
DATA_DIR = "data/cifar10"

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

    dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )

    loader = DataLoader(
        torch.utils.data.Subset(dataset, range(1000)),
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    student = StudentCNN(feature_dim=256).to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    student.load_state_dict(checkpoint["student_state_dict"])
    student.eval()

    all_features = []

    with torch.no_grad():
        for images, _ in loader:
            features = student(images)
            all_features.append(features)

    features = torch.cat(all_features, dim=0)

    normalized = F.normalize(features, p=2, dim=1)

    # Similarity between different images
    similarity_matrix = normalized @ normalized.T

    mask = ~torch.eye(
        len(features),
        dtype=torch.bool,
    )

    average_similarity = similarity_matrix[mask].mean()

    print("\n===== STUDENT FEATURE DIAGNOSTICS =====")
    print("Feature shape:", features.shape)
    print("Feature mean:", features.mean().item())
    print("Feature standard deviation:", features.std().item())
    print("Average pairwise cosine similarity:",
          average_similarity.item())


if __name__ == "__main__":
    main()
