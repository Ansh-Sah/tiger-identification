import torch
import torch.nn as nn
from torchvision import models


class TigerEmbeddingModel(nn.Module):

    def __init__(self):
        super().__init__()

        # ResNet50 backbone
        base_model = models.resnet50(weights=None)

        # Remove original classification layer
        self.backbone = nn.Sequential(
            *list(base_model.children())[:-1]
        )

        # 2048 → 512 → 256 embedding
        self.embedding_layer = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )

    def forward(self, x):

        # Extract ResNet features
        x = self.backbone(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Generate 256-dimensional embedding
        x = self.embedding_layer(x)

        # Normalize embedding
        x = torch.nn.functional.normalize(
            x,
            p=2,
            dim=1
        )

        return x