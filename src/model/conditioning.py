# embedding של הגל
import torch
import torch.nn as nn

class ContextEncoder(nn.Module):
    def __init__(self, input_dim, context_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, context_dim),
        )

    def forward(self, x):
        return self.net(x)
#