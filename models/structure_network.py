import torch
from torch import nn

from models.ipa_pytorch import InvariantPointAttention, BackboneUpdate
from models.structure_transition import StructureTransition

class StructureLayer(nn.Module):

    def __init__(self, module_cfg):
        super(StructureLayer, self).__init__()

        ipa_cfg = module_cfg.ipa
        c_s = ipa_cfg.c_s
        self.ipa = InvariantPointAttention(module_cfg.ipa)
        self.ipa_dropout = nn.Dropout(module_cfg.dropout)
        self.ipa_layer_norm = nn.LayerNorm(c_s)

        # Built-in dropout and layer norm
        self.transition = StructureTransition(
            c_s,
            module_cfg.n_structure_transition_layer, 
            module_cfg.structure_transition_dropout
        )
        
        # backbone update
        self.bb_update = BackboneUpdate(c_s)

    def forward(self, inputs):
        s, p, t, mask = inputs
        
        s = s + self.ipa(s, p, t, mask)
        s = self.ipa_dropout(s)
        s = self.ipa_layer_norm(s)
        s = self.transition(s)
        t = t.compose_q_update_vec(self.bb_update(s), mask[..., None])
        outputs = (s, p, t, mask)
        return outputs


class StructureNet(nn.Module):

    def __init__(self, module_cfg):
        super(StructureNet, self).__init__()
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * 0.1)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * 10.0)
        layers = [
            StructureLayer(module_cfg)
            for _ in range(module_cfg.n_structure_layer)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, s, p, t, mask):
        t = self.rigids_ang_to_nm(t)
        s, p, t, mask = self.net((s, p, t, mask))
        t = self.rigids_nm_to_ang(t)
        return t