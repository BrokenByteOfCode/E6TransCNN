import torch
import torch.nn as nn

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, xs, ys):
        xs_sig = torch.sigmoid(xs)
        
        targets = ys
        anti_targets = 1 - ys
        
        xs_pos = xs_sig
        xs_neg = 1 - xs_sig
        
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
            
        with torch.no_grad():
            asym_w_pos = (1 - xs_pos) ** self.gamma_pos
            asym_w_neg = (1 - xs_neg) ** self.gamma_neg
            
        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps)) * asym_w_pos
        loss_neg = anti_targets * torch.log(xs_neg.clamp(min=self.eps)) * asym_w_neg
        
        loss = -(loss_pos + loss_neg)
        return loss.mean()