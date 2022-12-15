import torch
from torch import nn
import torch.nn.functional as F


class PoseEncoderFourierMLP(nn.Module):
    """
    Inspired from positional encodings in NeRF
    """

    def __init__(self, P_embed, d_model, n_input_dim):
        super().__init__()
        self.P_embed = P_embed
        self.d_model = d_model
        self.n_input_dim = n_input_dim
        self.pose_encoding_size = self.n_input_dim * (1 + 2 * P_embed)
        self.ff = nn.Sequential(
            nn.Linear(self.pose_encoding_size, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, self.d_model),
        )

    def forward(self, position):
        rets = [position]
        for i in range(self.P_embed):
            for fn in [torch.sin, torch.cos]:
                rets.append(fn(2.0**i * position))
        out = torch.cat(rets, dim=-1)
        return self.ff(out)
