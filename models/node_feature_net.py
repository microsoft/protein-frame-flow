import torch
from torch import nn
from models.utils import get_index_embedding, get_time_embedding


class NodeFeatureNet(nn.Module):

    def __init__(self, module_cfg):
        super(NodeFeatureNet, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb

        embed_size = self._cfg.c_pos_emb
        embed_size += self._cfg.c_timestep_emb
        if 'separate_t' not in self._cfg:
            self._separate_t = False
        else:
            self._separate_t = self._cfg.separate_t
        if self._separate_t:
            embed_size += self._cfg.c_timestep_emb
        if self._cfg.embed_diffuse_mask:
            embed_size += 1
        self.linear = nn.Linear(embed_size, self.c_s)

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.c_timestep_emb,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(self, timesteps, so3_t, r3_t, mask):
        # s: [b]

        b, num_res, device = mask.shape[0], mask.shape[1], mask.device

        # [b, n_res, c_pos_emb]
        pos = torch.arange(num_res, dtype=torch.float32).to(device)[None]
        pos_emb = get_index_embedding(
            pos, self.c_pos_emb, max_len=2056
        )
        pos_emb = pos_emb.repeat([b, 1, 1])
        pos_emb = pos_emb * mask.unsqueeze(-1)

        # [b, n_res, c_timestep_emb]
        input_feats = [pos_emb]
        if self._cfg.embed_diffuse_mask:
            input_feats.append(mask[..., None])
        # timesteps are between 0 and 1. Convert to integers.
        if self._separate_t:
            input_feats.append(self.embed_t(so3_t, mask))
            input_feats.append(self.embed_t(r3_t, mask))
        else:
            input_feats.append(self.embed_t(timesteps, mask))
        return self.linear(torch.cat(input_feats, dim=-1))