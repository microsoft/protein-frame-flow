from torch import nn

from models.pair_transition import PairTransition
from models.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)


class PairTransformLayer(nn.Module):

    def __init__(self, module_cfg):
        super(PairTransformLayer, self).__init__()
        
        self._cfg = module_cfg
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

    def forward(self, inputs):
        p, p_mask = inputs
        p = p + self.tri_mul_out(p, p_mask)
        p = p + self.tri_mul_in(p, p_mask)
        p = p + self.pair_transition(p, p_mask)
        p = p * p_mask.unsqueeze(-1)
        outputs = (p, p_mask)
        return outputs

class PairTransformNet(nn.Module):

    def __init__(self, module_cfg):
        super(PairTransformNet, self).__init__()

        layers = [
            PairTransformLayer(module_cfg)
            for _ in range(module_cfg.n_pair_transform_layer)
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, p, p_mask):
        p, _ = self.net((p, p_mask))
        return p