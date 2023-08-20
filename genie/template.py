import torch

from genie.utils.geo_utils import distance, dihedral


def get_template_fn(template):
	if template == 'v1':
		return v1, 1

def v1(t):

	# [b, n_res, n_res, 1]
	d = distance(torch.stack([
		t.trans.unsqueeze(2).repeat(1, 1, t.trans.shape[1], 1), # Ca_1
		t.trans.unsqueeze(1).repeat(1, t.trans.shape[1], 1, 1), # Ca_2
	], dim=-2)).unsqueeze(-1)

	return d