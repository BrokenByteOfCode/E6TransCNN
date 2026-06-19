import torch
import torch.nn as nn

class RatingHead(nn.Module):
    def __init__(self, embed_dim=768, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

class GenderHead(nn.Module):
    def __init__(self, embed_dim=768, num_classes=4):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

class GeneralHead(nn.Module):
    def __init__(self, embed_dim=768, num_classes=500):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

class MetaHead(nn.Module):
    def __init__(self, embed_dim=768, num_classes=50):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)