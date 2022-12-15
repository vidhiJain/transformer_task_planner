from turtle import forward
import torch
from torch import nn


class CategoryEncoderFourierMLP(nn.Module):
    """Takes Bounding box extents in object's body frame
     and form category specific embed

    MLP to process the fourier decomposition
    """

    def __init__(self, C_embed, h_dim):
        super().__init__()
        self.C_embed = C_embed
        self.h_dim = h_dim
        self.category_encoder_size = 3 * (1 + 2 * C_embed)
        self.ff = nn.Sequential(
            nn.Linear(self.category_encoder_size, h_dim // 2),
            nn.ReLU(),
            nn.Linear(h_dim // 2, h_dim),
        )

    def forward(self, bb):
        rets = [bb]
        for i in range(self.C_embed):
            for fn in [torch.sin, torch.cos]:
                rets.append(fn(2.0**i * bb))
        out = torch.cat(rets, dim=-1)
        return self.ff(out)


class CategoryEncoderFourier(CategoryEncoderFourierMLP):
    """
    Fourier only, No learned aspect!
    """

    def __init__(self, C_embed, h_dim):
        super().__init__()

    def forward(self, bb):
        rets = [bb]
        for i in range(self.C_embed):
            for fn in [torch.sin, torch.cos]:
                rets.append(fn(2.0**i * bb))
        out = torch.cat(rets, dim=-1)
        return out[: self.h_dim]


class CategoryEncoderMLP(CategoryEncoderFourierMLP):
    """
    only MLP, no fourier decomposition!
    """

    def __init__(self, h_dim):
        super().__init__(C_embed=0, h_dim=h_dim)

    def forward(self, bb):
        return self.ff(bb)
