import torch
from torch import nn

from genie.model.modules.invariant_point_attention import InvariantPointAttention
from genie.model.modules.structure_transition import StructureTransition
from genie.model.modules.backbone_update import BackboneUpdate


class StructureLayer(nn.Module):

	def __init__(self,
		c_s,
		c_p,
		c_hidden_ipa,
		n_head,
		n_qk_point,
		n_v_point,
		ipa_dropout,
		n_structure_transition_layer,
		structure_transition_dropout
	):
		super(StructureLayer, self).__init__()

		self.ipa = InvariantPointAttention(
			c_s,
			c_p,
			c_hidden_ipa,
			n_head,
			n_qk_point,
			n_v_point
		)
		self.ipa_dropout = nn.Dropout(ipa_dropout)
		self.ipa_layer_norm = nn.LayerNorm(c_s)

		# Built-in dropout and layer norm
		self.transition = StructureTransition(
			c_s,
			n_structure_transition_layer, 
			structure_transition_dropout
		)
		
		# backbone update
		self.bb_update = BackboneUpdate(c_s)

	def forward(self, inputs):
		s, p, t, mask = inputs
		s = s + self.ipa(s, p, t, mask)
		s = self.ipa_dropout(s)
		s = self.ipa_layer_norm(s)
		s = self.transition(s)
		t = t.compose(self.bb_update(s))
		outputs = (s, p, t, mask)
		return outputs


class StructureNet(nn.Module):

	def __init__(self,
		c_s,
		c_p,
		n_structure_layer,
		n_structure_block,
		c_hidden_ipa,
		n_head_ipa,
		n_qk_point,
		n_v_point,
		ipa_dropout,
		n_structure_transition_layer,
		structure_transition_dropout		
	):
		super(StructureNet, self).__init__()

		self.n_structure_block = n_structure_block

		layers = [
			StructureLayer(
				c_s, c_p,
				c_hidden_ipa, n_head_ipa, n_qk_point, n_v_point, ipa_dropout, 
				n_structure_transition_layer, structure_transition_dropout
			)
			for _ in range(n_structure_layer)
		]
		self.net = nn.Sequential(*layers)

	def forward(self, s, p, t, mask):
		for block_idx in range(self.n_structure_block):
			s, p, t, mask = self.net((s, p, t, mask))
		return t