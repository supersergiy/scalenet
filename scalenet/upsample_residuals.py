import torch

class UpsampleResiduals(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, resdidual, scale=2, **kwargs):
        res = res.permute(0, 3, 1, 2)
        result = nn.functional.interpolate(res, scale_factor=scale)
        result *= scale
        result = result.permute(0, 2, 3, 1)

        return result

