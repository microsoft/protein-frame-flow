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

class StructureUpdate(nn.Module):

    def __init__(self, module_cfg):
        super(StructureUpdate, self).__init__()
        self._module_cfg = module_cfg
        ipa_cfg = module_cfg.ipa
        c_s = ipa_cfg.c_s
        self.ipa = InvariantPointAttention(module_cfg.ipa)
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

    def forward(self, p, p_mask):
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
        for b in range(self._cfg.num_blocks):
            self.trunk[f'edge_update_{b}'] = EdgeUpdate(
                self._cfg.edge_update) 
            self.trunk[f'bb_update_{b}'] = StructureUpdate(
                self._cfg.structure_update)

    def forward(self, s, p, t, mask, p_mask):
        t = self.rigids_ang_to_nm(t)
        for b in range(self._cfg.num_blocks):
            p = self.trunk[f'edge_update_{b}'](p, p_mask)
            s, t = self.trunk[f'bb_update_{b}'](s, p, t, mask)
        t = self.rigids_nm_to_ang(t)
        return s, t