from torch import nn

from genie.model.modules.pair_transition import PairTransition
from genie.model.modules.triangular_attention import (
	TriangleAttentionStartingNode,
	TriangleAttentionEndingNode,
)
from genie.model.modules.triangular_multiplicative_update import (
	TriangleMultiplicationOutgoing,
	TriangleMultiplicationIncoming,
)
from genie.model.modules.dropout import (
	DropoutRowwise,
	DropoutColumnwise
)


class PairTransformLayer(nn.Module):

	def __init__(self,
		c_p,
		include_mul_update,
		include_tri_att,
		c_hidden_mul,
		c_hidden_tri_att,
		n_head_tri,
		tri_dropout,
		pair_transition_n,
	):
		super(PairTransformLayer, self).__init__()

		self.tri_mul_out = TriangleMultiplicationOutgoing(
			c_p,
			c_hidden_mul
		) if include_mul_update else None

		self.tri_mul_in = TriangleMultiplicationIncoming(
			c_p,
			c_hidden_mul
		) if include_mul_update else None

		self.tri_att_start = TriangleAttentionStartingNode(
			c_p,
			c_hidden_tri_att,
			n_head_tri
		) if include_tri_att else None

		self.tri_att_end = TriangleAttentionEndingNode(
			c_p,
			c_hidden_tri_att,
			n_head_tri
		) if include_tri_att else None

		self.pair_transition = PairTransition(
			c_p,
			pair_transition_n
		)

		self.dropout_row_layer = DropoutRowwise(tri_dropout)
		self.dropout_col_layer = DropoutColumnwise(tri_dropout)

	def forward(self, inputs):
		p, p_mask = inputs
		if self.tri_mul_out is not None:
			p = p + self.dropout_row_layer(self.tri_mul_out(p, p_mask))
			p = p + self.dropout_row_layer(self.tri_mul_in(p, p_mask))
		if self.tri_att_start is not None:
			p = p + self.dropout_row_layer(self.tri_att_start(p, p_mask))
			p = p + self.dropout_col_layer(self.tri_att_end(p, p_mask))
		p = p + self.pair_transition(p, p_mask)
		p = p * p_mask.unsqueeze(-1)
		outputs = (p, p_mask)
		return outputs

class PairTransformNet(nn.Module):

	def __init__(self,
		c_p,
		n_pair_transform_layer,
		include_mul_update,
		include_tri_att,
		c_hidden_mul,
		c_hidden_tri_att,
		n_head_tri,
		tri_dropout,
		pair_transition_n
	):
		super(PairTransformNet, self).__init__()

		layers = [
			PairTransformLayer(
				c_p,
				include_mul_update,
				include_tri_att,
				c_hidden_mul,
				c_hidden_tri_att,
				n_head_tri,
				tri_dropout,
				pair_transition_n
			)
			for _ in range(n_pair_transform_layer)
		]

		self.net = nn.Sequential(*layers)

	def forward(self, p, p_mask):
		p, _ = self.net((p, p_mask))
		return p