import torch

class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, **kwargs):
        return x
