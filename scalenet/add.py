import torch

class Add(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, **kwargs):
        return x + y
