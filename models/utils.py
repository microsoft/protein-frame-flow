import math
import torch


def sinusoidal_encoding(v, N, D):
	"""Taken from GENIE.
	
	Args:

	"""
	# v: [*]

	# [D]
	k = torch.arange(1, D+1).to(v.device)

	# [*, D]
	sin_div_term = N ** (2 * k / D)
	sin_div_term = sin_div_term.view(*((1, ) * len(v.shape) + (len(sin_div_term), )))
	sin_enc = torch.sin(v.unsqueeze(-1) * math.pi / sin_div_term)

	# [*, D]
	cos_div_term = N ** (2 * (k - 1) / D)
	cos_div_term = cos_div_term.view(*((1, ) * len(v.shape) + (len(cos_div_term), )))
	cos_enc = torch.cos(v.unsqueeze(-1) * math.pi / cos_div_term)

	# [*, D]
	enc = torch.zeros_like(sin_enc).to(v.device)
	enc[..., 0::2] = cos_enc[..., 0::2]
	enc[..., 1::2] = sin_enc[..., 1::2]

	return enc.to(v.dtype)


def distance(p, eps=1e-10):
    # [*, 2, 3]
    return (eps + torch.sum((p[..., 0, :] - p[..., 1, :]) ** 2, dim=-1)) ** 0.5


def dist_from_ca(trans):

	# [b, n_res, n_res, 1]
	d = distance(torch.stack([
		trans.unsqueeze(2).repeat(1, 1, trans.shape[1], 1), # Ca_1
		trans.unsqueeze(1).repeat(1, trans.shape[1], 1, 1), # Ca_2
	], dim=-2)).unsqueeze(-1)

	return d

