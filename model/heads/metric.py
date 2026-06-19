import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeciesMetricHead(nn.Module):
    def __init__(self, embed_dim=768, num_classes=100, init_scale=30.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        self.scale = nn.Parameter(torch.tensor(init_scale))
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=-1)
        w_norm = F.normalize(self.weight, p=2, dim=-1)
        
        cosine = F.linear(x_norm, w_norm)
        logits = cosine * self.scale
        return logits