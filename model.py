import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """A two-layer neural network."""
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(1, 40)
        self.fc1 = nn.Linear(40, 40)
        self.fc2 = nn.Linear(40, 1)
    
    def forward(self, x, params=None):
        if params is None:
            params = list(self.parameters())
        x = F.linear(x, params[0], params[1])
        x = F.relu(x)
        x = F.linear(x, params[2], params[3])
        x = F.relu(x)
        x = F.linear(x, params[4], params[5])
        return x