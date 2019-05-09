import torch
import warnings

class UpsampleResiduals(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, res, scale=2, **kwargs):
        if res.shape[3] == 2 and res.shape[1] == 2:
            warnings.warn("Ambiguous Residual: both 2nd and 4th dimensions of residual are 2.")

        if res.shape[3] != 2 and res.shape[1] == 2:
            channel_permute = False
        else:
            channel_permute = True

        if channel_permute:
            res = res.permute(0, 3, 1, 2)

        result = torch.nn.functional.interpolate(res, scale_factor=scale)
        result *= scale

        if channel_permute:
            result = result.permute(0, 2, 3, 1)

        return result

