import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        
        # TODO
        # Compute mean
        mean = x.mean(dim=-1, keepdim=True)

        # Compute variance
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        normalized_x = (x - mean) / torch.sqrt(variance + 1e-10)
        
        # Apply scale and shift
        return self.scale * normalized_x + self.shift