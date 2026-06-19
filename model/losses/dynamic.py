import torch
import torch.nn as nn

class UncertaintyLossBalancer(nn.Module):
    def __init__(self, num_tasks=5):
        super().__init__()
        self.num_tasks = num_tasks
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        assert len(losses) == self.num_tasks
        
        total_loss = 0.0
        for i, loss in enumerate(losses):
            log_var_clamped = self.log_vars[i].clamp(min=-5.0, max=5.0)
            precision = torch.exp(-log_var_clamped)
            total_loss += precision * loss + log_var_clamped
            
        return total_loss