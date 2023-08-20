import torch
from torch import nn

from models.utils import dist_from_ca

class EdgeFeatureNet(nn.Module):

	def __init__(self, module_cfg):
		#   c_s, c_p, relpos_k, template_type):
		super(EdgeFeatureNet, self).__init__()
		self._cfg = module_cfg

		self.c_s = self._cfg.c_s
		self.c_p = self._cfg.c_p

		self.linear_s_p_i = nn.Linear(self.c_s, self.c_p)
		self.linear_s_p_j = nn.Linear(self.c_s, self.c_p)

		self.relpos_k = self._cfg.relpos_k
		self.n_bin = 2 * self._cfg.relpos_k + 1
		self.linear_relpos = nn.Linear(self.n_bin, self.c_p)
		self.linear_template = nn.Linear(1, self.c_p)

	def relpos(self, r):
		# AlphaFold 2 Algorithm 4 & 5
		# Based on OpenFold utils/tensor_utils.py
		# Input: [b, n_res]

		# [b, n_res, n_res]
		d = r[:, :, None] - r[:, None, :]

		# [n_bin]
		v = torch.arange(-self.relpos_k, self.relpos_k + 1).to(r.device)
		
		# [1, 1, 1, n_bin]
		v_reshaped = v.view(*((1,) * len(d.shape) + (len(v),)))

		# [b, n_res, n_res]
		b = torch.argmin(torch.abs(d[:, :, :, None] - v_reshaped), dim=-1)

		# [b, n_res, n_res, n_bin]
		oh = nn.functional.one_hot(b, num_classes=len(v)).float()

		# [b, n_res, n_res, c_p]
		p = self.linear_relpos(oh)

		return p

	def forward(self, s, t, p_mask):
		# Input: [b, n_res, c_s]

		# [b, n_res, c_p]
		p_i = self.linear_s_p_i(s)
		p_j = self.linear_s_p_j(s)

		# [b, n_res, n_res, c_p]
		p = p_i[:, :, None, :] + p_j[:, None, :, :]

		# [b, n_res]
		r = torch.arange(s.shape[1]).unsqueeze(0).repeat(s.shape[0], 1).to(s.device)

		# [b, n_res, n_res, c_p]
		p += self.relpos(r)
		p += self.linear_template(dist_from_ca(t))

		# [b, n_res, n_res, c_p]
		p *= p_mask.unsqueeze(-1)

		return p