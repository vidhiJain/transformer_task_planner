import torch
from torch import nn


class TemporalEncoderFourierMLP(nn.Module):
    """
    Inspired from positional encodings in Transformers (Vaswani et al)
    """

    def __init__(self, T_embed, d_model):
        super().__init__()
        self.T_embed = T_embed
        self.d_model = d_model
        self.temporal_encoding_size = 2 * T_embed
        self.ff = nn.Sequential(
            nn.Linear(self.temporal_encoding_size, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, self.d_model),
        )

    def forward(self, timestep):
        rets = []
        for i in range(self.T_embed):
            for fn in [torch.sin, torch.cos]:
                rets.append(fn(2.0**i * timestep))
        out = torch.cat(rets, dim=-1)
        return self.ff(out)


class TemporalEncoderEmbedding(nn.Module):
    """
    embedding layer to learn the timestep
    """

    def __init__(self, T_embed, d_model):
        super().__init__()
        self.d_model = d_model
        self.temporal_encoding_size = T_embed + 1
        self.embed_layer = nn.Embedding(self.temporal_encoding_size, d_model)

    def forward(self, timestep):
        return self.embed_layer(timestep.long())
