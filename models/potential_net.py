import torch
from torch import nn

from models.ipa_pytorch import InvariantPointAttention, BackboneUpdate
from models.structure_transition import StructureTransition
from models.pair_transition import PairTransition
from models.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
    SymmetricTriangleUpdate,
)
from models.utils import calc_rbf, dist_from_ca
from models.primitives import Linear

class StructureUpdate(nn.Module):

    def __init__(self, module_cfg):
        super(StructureUpdate, self).__init__()
        self._module_cfg = module_cfg
        ipa_cfg = module_cfg.ipa
        c_s = ipa_cfg.c_s
        self.ipa = InvariantPointAttention(module_cfg.ipa)
        seq_tfmr_cfg = module_cfg.seq_tfmr
        if seq_tfmr_cfg.enable:
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=c_s,
                nhead=seq_tfmr_cfg.num_heads,
                dim_feedforward=c_s,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.seq_tfmr = torch.nn.TransformerEncoder(
                tfmr_layer,
                seq_tfmr_cfg.num_layers,
                enable_nested_tensor=False
            )
            self.seq_tfmr_layer_norm = nn.LayerNorm(c_s)
        self.ipa_dropout = nn.Dropout(module_cfg.dropout)
        self.ipa_layer_norm = nn.LayerNorm(c_s)

        # Built-in dropout and layer norm
        self.transition = StructureTransition(
            c_s,
            module_cfg.num_transition_layers, 
            module_cfg.dropout
        )
        
        # backbone update
        self.bb_update = BackboneUpdate(
            c_s, module_cfg.use_rot_updates)

    def forward(self, s, p, t, mask):
        s = s + self.ipa(s, p, t, mask)
        s = self.ipa_dropout(s)
        s = self.ipa_layer_norm(s)
        s = s * mask[..., None]
        if self._module_cfg.seq_tfmr.enable:
            s = s + self.seq_tfmr(s, src_key_padding_mask=(1 - mask).type(torch.float32))
            s = self.seq_tfmr_layer_norm(s)
        s = self.transition(s)
        if self._module_cfg.use_rot_updates:
            t = t.compose_q_update_vec(self.bb_update(s), mask[..., None])
        else:
            t = t.compose_tran_update_vec(self.bb_update(s), mask[..., None])
        return s, t


class EdgeUpdate(nn.Module):

    def __init__(self, module_cfg):
        super(EdgeUpdate, self).__init__()
        
        self._cfg = module_cfg
        if self._cfg.single_bias:
            self.single_a =  Linear(self._cfg.c_s, self._cfg.c_s)
            self.single_a_ln = nn.LayerNorm(self._cfg.c_s)
            self.final_a = nn.Sequential(
                Linear(
                    self._cfg.c_s + self._cfg.num_rbf, self._cfg.c_p,
                    init="relu"),
                nn.ReLU(),
                Linear(self._cfg.c_p, self._cfg.c_p, init="relu"),
                nn.ReLU(),
                Linear(self._cfg.c_p, self._cfg.c_p),
            )
        if self._cfg.symmetric_update:
            self.symm_tri_mul = SymmetricTriangleUpdate(
                self._cfg.c_p,
                self._cfg.c_hidden_mul
            )
        else:
            self.tri_mul_out = TriangleMultiplicationOutgoing(
                self._cfg.c_p,
                self._cfg.c_hidden_mul
            )

            self.tri_mul_in = TriangleMultiplicationIncoming(
                self._cfg.c_p,
                self._cfg.c_hidden_mul
            )

        self.pair_transition = PairTransition(
            self._cfg.c_p,
            self._cfg.pair_transition_n
        )

    def forward(self, s, p, t, p_mask):
        if self._cfg.single_bias:
            num_batch, num_res = s.shape[:2]
            device = p_mask.device
            single_bias = self.single_a(s)
            a_ij = self.single_a_ln(
                single_bias[:, :, None, :] * single_bias[:, None, :, :]
            )
            rbf_feats = calc_rbf(
                dist_from_ca(t.get_trans()), self._cfg.num_rbf)
            single_bias_feats = torch.concat([a_ij, rbf_feats], dim=-1)
            all_ones = torch.ones(num_res, num_res, device=device)
            triu_idx = torch.where(
                torch.triu(all_ones, diagonal=1)[None].repeat(num_batch, 1, 1))
            sparse_off_diag_update = self.final_a(single_bias_feats[triu_idx])

            p_bias = torch.zeros_like(p)
            p_bias[triu_idx] = sparse_off_diag_update
            p_bias = p_bias + p_bias.swapaxes(1, 2) + torch.eye(num_res, device=device)[None, ..., None] * s[:, :, None, :].repeat(1, 1, num_res, 1)
            p = p + p_bias
        if self._cfg.symmetric_update:
            symm_update = self.symm_tri_mul(p, p_mask) 
            p = p + symm_update
        else:
            p = p + self.tri_mul_out(p, p_mask)
            p = p + self.tri_mul_in(p, p_mask)
        p = p + self.pair_transition(p, p_mask)
        p = p * p_mask.unsqueeze(-1)
        return p

class PotentialNet(nn.Module):

    def __init__(self, module_cfg):
        super(PotentialNet, self).__init__()
        self._cfg = module_cfg
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * 0.1)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * 10.0)
        self.trunk = nn.ModuleDict()
        if self._cfg.weight_share:
            edge_update = EdgeUpdate(self._cfg.edge_update)
            bb_update = StructureUpdate(self._cfg.structure_update)
        for b in range(self._cfg.num_blocks):
            if self._cfg.weight_share:
                self.trunk[f'edge_update_{b}'] = edge_update
                self.trunk[f'bb_update_{b}'] = bb_update
            else:
                self.trunk[f'edge_update_{b}'] = EdgeUpdate(
                    self._cfg.edge_update)
                self.trunk[f'bb_update_{b}'] = StructureUpdate(
                    self._cfg.structure_update)

    def forward(self, s, p, t, mask, p_mask):
        t = self.rigids_ang_to_nm(t)
        for b in range(self._cfg.num_blocks):
            p = self.trunk[f'edge_update_{b}'](s, p, t, p_mask)
            s, t = self.trunk[f'bb_update_{b}'](s, p, t, mask)
        t = self.rigids_nm_to_ang(t)
        return s, t