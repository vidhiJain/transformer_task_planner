from typing import Any, Dict
import numpy as np
import torch
from torch import nn
from torch.nn import (
    TransformerEncoderLayer,
    TransformerEncoder,
    TransformerDecoderLayer,
    TransformerDecoder,
)
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig


def generate_square_subsequent_mask(sz: int, diagonal=1):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    return torch.triu(torch.full((sz, sz), float("-inf")), diagonal=diagonal)


class TransformerTaskPlannerDualModel(PreTrainedModel):
    """Encoding for category, pose, timesteps and reality marker
    with pick and place decoding heads
    """

    def __init__(
        self,
        config: PretrainedConfig,
        category_encoder: torch.nn.Module,
        pose_encoder: torch.nn.Module,
        temporal_encoder: torch.nn.Module,
        reality_marker_encoder: torch.nn.Module,
    ):
        super().__init__(config)
        self.category_encoder = category_encoder
        self.pose_encoder = pose_encoder
        self.temporal_encoder = temporal_encoder
        self.reality_marker_encoder = reality_marker_encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.d_hid,
            dropout=config.dropout,
            batch_first=config.batch_first,
        )
        self.pick_encoder = TransformerEncoder(
            encoder_layers, config.num_encoder_layers
        )
        decoder_layers = TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.d_hid,
            dropout=config.dropout,
            batch_first=config.batch_first,
        )
        self.place_decoder = TransformerDecoder(
            decoder_layers, config.num_decoder_layers
        )

    def forward(
        self,
        rigid_instances,
        place_instances,
        src_key_padding_mask,
        device,
    ):
        """
        Args:
            # timestep: (B, N, 1) long tensor
            # category (bounding box max extents): (B, N, 3) float tensor
            # pose: (B, N, 7) float tensor
            # action_masks: (B, N, 1) bool tensor
            # is_real: (B, N, 1) bool tensor
            # category_token: (B, N, 1), long tensor
            # instance_token: (B, N, 1), long tensor
            rigid_instances
            place_instances
            src_key_padding_mask: (B, N, 1) bool tensor
            device: str ('cpu' / 'cuda')
        Return:
            pick_id : (A, N)
            feasible_placements : (A, T)  (sigmoid over each output)
                where A is the number of ACT tokens across the batch, N are the num of instances
        """
        src_mask = generate_square_subsequent_mask(pose.shape[1]).to(device)
        bs = pose.shape[0]
        msl = pose.shape[1]
        x_t = self.temporal_encoder(timestep.reshape(-1, 1).to(device))
        x_t = x_t.view(bs, msl, self.config.temporal_embed_size)

        x_c = self.category_encoder(category.reshape(-1, 3).to(device))
        x_c = x_c.view(bs, msl, self.config.category_embed_size)

        position = pose[:, :, : self.config.n_input_dim]
        x_p = self.pose_encoder(
            position.reshape(-1, self.config.n_input_dim).to(device)
        )
        x_p = x_p.view(bs, msl, self.config.pose_embed_size)

        x_m = self.reality_marker_encoder(is_real.long().reshape(-1, 1).to(device))
        x_m = x_m.view(bs, msl, self.config.marker_embed_size)

        x = torch.cat([x_m, x_t, x_c, x_p], dim=-1)
        memory = self.transformer_encoder(
            x, src_mask.to(device), src_key_padding_mask.to(device)
        )
        raw_out_pick = memory @ x.permute(0, 2, 1) + generate_square_subsequent_mask(
            pose.shape[1], diagonal=0
        ).to(device)
        out_pick = raw_out_pick[action_masks, :]
        out_place = None
        out = {"pick": out_pick, "place": out_place}
        return out


if __name__ == "__main__":
    model = TransformerTaskPlannerDualModel(
        config, category_encoder, pose_encoder, temporal_encoder, reality_marker_encoder
    )
