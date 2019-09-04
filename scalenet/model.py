import sys
import os

import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.best_val = sys.maxsize
        self.name = "default"
        self.params = {}

    def forward(self, x):
        return x

    def save_state_dict(self, checkpoint_folder='./checkpoints/'):
        path = os.path.join(checkpoint_folder, '{}.state.pth.tar'.format(self.name))
        torch.save(self.state_dict(), path)

    def save_nonportable(self, checkpoint_folder='./checkpoints/'):
        path = os.path.join(checkpoint_folder, '{}.pth.tar'.format(self.name))
        torch.save({'model': self}, path)

    def get_all_params(self):
        result = []
        result.extend(self.parameters())
        return result


