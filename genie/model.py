import torch
from torch import nn

from genie.model.single_feature_net import SingleFeatureNet
from genie.model.pair_feature_net import PairFeatureNet
from genie.model.pair_transform_net import PairTransformNet
from genie.model.structure_net import StructureNet


class Denoiser(nn.Module):

	def __init__(self,
		c_s, c_p, n_timestep,
		c_pos_emb, c_timestep_emb,
		relpos_k, template_type,
		n_pair_transform_layer, include_mul_update, include_tri_att,
		c_hidden_mul, c_hidden_tri_att, n_head_tri, tri_dropout, pair_transition_n,
		n_structure_layer, n_structure_block,
		c_hidden_ipa, n_head_ipa, n_qk_point, n_v_point, ipa_dropout,
		n_structure_transition_layer, structure_transition_dropout
	):
		super(Denoiser, self).__init__()

		self.single_feature_net = SingleFeatureNet(
			c_s,
			n_timestep,
			c_pos_emb,
			c_timestep_emb
		)
		
		self.pair_feature_net = PairFeatureNet(
			c_s,
			c_p,
			relpos_k,
			template_type
		)

		self.pair_transform_net = PairTransformNet(
			c_p,
			n_pair_transform_layer,
			include_mul_update,
			include_tri_att,
			c_hidden_mul,
			c_hidden_tri_att,
			n_head_tri,
			tri_dropout,
			pair_transition_n
		) if n_pair_transform_layer > 0 else None

		self.structure_net = StructureNet(
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
		)

	def forward(self, ts, timesteps, mask):
		p_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
		s = self.single_feature_net(ts, timesteps, mask)
		p = self.pair_feature_net(s, ts, p_mask)
		if self.pair_transform_net is not None:
			p = self.pair_transform_net(p, p_mask)
		ts = self.structure_net(s, p, ts, mask)
		return ts