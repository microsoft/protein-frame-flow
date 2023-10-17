import math
import torch
from torch.nn import functional as F
import numpy as np
from data import utils as du


def calc_distogram(pos, min_bin, max_bin, num_bins):
    dists_2d = torch.linalg.norm(
        pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[..., None]
    lower = torch.linspace(
        min_bin,
        max_bin,
        num_bins,
        device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return dgram


def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size//2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def get_time_embedding(timesteps, embedding_dim, max_positions=2000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


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


def calc_rbf(ca_dists, num_rbf, D_min=1e-3, D_max=22.):
    # Distance radial basis function
    device = ca_dists.device
    D_mu = torch.linspace(D_min, D_max, num_rbf).to(device)
    D_mu = D_mu.view([1,1,1,-1])
    D_sigma = (D_max - D_min) / num_rbf
    return torch.exp(-((ca_dists - D_mu) / D_sigma)**2)


def t_stratified_loss(batch_t, batch_loss, num_bins=4, loss_name=None):
    """Stratify loss by binning t."""
    batch_t = du.to_numpy(batch_t)
    batch_loss = du.to_numpy(batch_loss)
    flat_losses = batch_loss.flatten()
    flat_t = batch_t.flatten()
    bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins+1)
    bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
    t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
    t_binned_n = np.bincount(bin_idx)
    stratified_losses = {}
    if loss_name is None:
        loss_name = 'loss'
    for t_bin in np.unique(bin_idx).tolist():
        bin_start = bin_edges[t_bin]
        bin_end = bin_edges[t_bin+1]
        t_range = f'{loss_name} t=[{bin_start:.2f},{bin_end:.2f})'
        range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
        stratified_losses[t_range] = range_loss
    return stratified_losses
