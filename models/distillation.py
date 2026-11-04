import torch.nn as nn


class TeacherProjection(nn.Module):
    def __init__(self, teacher_dim=512, student_dim=256):
        super().__init__()
        self.projection = nn.Linear(teacher_dim, student_dim)

    def forward(self, teacher_features):
        return self.projection(teacher_features)