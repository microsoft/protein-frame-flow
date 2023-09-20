import torch
from torch import nn

from models.utils import sinusoidal_encoding

class NodeFeatureNet(nn.Module):

    def __init__(self, module_cfg):
        super(NodeFeatureNet, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb

        embed_size = self._cfg.c_pos_emb
        if self._cfg.embed_t:
            embed_size += self._cfg.c_timestep_emb
        self.linear = nn.Linear(embed_size, self.c_s)

    def forward(self, timesteps, mask):
        # s: [b]

        b, num_res, device = mask.shape[0], mask.shape[1], timesteps.device

        # [b, n_res, c_pos_emb]
        pos_emb = sinusoidal_encoding(
            torch.arange(num_res, dtype=torch.float32).to(device)[None],
            self._cfg.max_num_res,
            self.c_pos_emb
        )
        pos_emb = pos_emb.repeat([b, 1, 1])
        pos_emb = pos_emb * mask.unsqueeze(-1)

        # [b, n_res, c_timestep_emb]
        # timesteps are between 0 and 1. Convert to integers.
        input_feats = [pos_emb]
        if self._cfg.embed_t:
            timesteps_int = torch.floor(
                timesteps * self._cfg.timestep_int).to(device)
            timestep_emb = sinusoidal_encoding(
                timesteps_int.view(b, 1),
                self._cfg.timestep_int,	
                self.c_timestep_emb
            )
            timestep_emb = timestep_emb.repeat(1, num_res, 1)
            timestep_emb = timestep_emb * mask.unsqueeze(-1)
            input_feats.append(timestep_emb)
        return self.linear(torch.cat(input_feats, dim=-1))