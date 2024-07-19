# from https://github1s.com/blt2114/twisted_diffusion_sampler/blob/main/protein_exp/motif_scaffolding/twisting.py

import math
import time
from operator import itemgetter
from itertools import groupby

import torch
from einops import rearrange
from openfold.utils.rigid_utils import Rigid, Rotation
from data import so3_utils
from data import all_atom
from data import utils as du
from analysis import utils as au
import numpy as np


def find_ranges_and_lengths(data):
    ranges, lengths = [], []
    for _, group in groupby(enumerate(data), lambda indexitem: indexitem[0] - indexitem[1]):
        group = list(map(itemgetter(1), group))
        lengths.append(group[-1] - group[0] + 1)
        ranges.append((group[0], group[-1]))
    return ranges, lengths


def perturbations_for_grad(sample_feats):
    """perturbations_for_grad turns xt composed with an identity perturbation
    from which gradients may be computed.

    Args:
        sample_feats: dict, contains sample features, including xt and Rt which are perturbed
        se3_diffuser: SE3Diffuser, used to scale and unscale xt
    """
    device = sample_feats['rotmats_t'].device
    #NOTE: otherwise getting 'RuntimeError: Inference tensors cannot be saved for backward.'
    for key, value in sample_feats.items():
        sample_feats[key] = value.clone().detach().requires_grad_(False)
    Rt = sample_feats['rotmats_t']#.clone().detach().requires_grad_(True)
    xt = sample_feats['trans_t']#.clone().detach().requires_grad_(True)

    delta_x = torch.zeros_like(xt, requires_grad=True)
    # xt = se3_diffuser._r3_diffuser._scale(xt)
    xt = xt + delta_x
    # xt = se3_diffuser._r3_diffuser._unscale(xt)

    Log_delta_R = torch.zeros_like(xt, requires_grad=True)
    skew_sym = so3_utils.hat(Log_delta_R)
    # delta_R = torch.einsum('...ij,...jk->...ik', Rt, so3_utils.hat(Log_delta_R))
    # Rt = so3_utils.expmap(Rt, delta_R)
    Rt = so3_utils.expmap(Rt, skew_sym=skew_sym)

    xt = xt.to(device)
    Rt = Rt.to(device)

    # update rigids_t to include perturbed Rt and xt
    # rigids_t = Rigid(rots=Rotation(rot_mats=Rt), trans=xt).to_tensor_7()

    sample_feats['trans_t'] = xt
    sample_feats['rotmats_t'] = Rt
    # sample_feats['R_t'] = Rt
    # sample_feats['rigids_t'] = rigids_t

    return sample_feats, Log_delta_R, delta_x


def step(trans_t, rotmats_t, trans_grad_t, grad_Log_delta_R, d_t, trans_scale_t, rot_scale_t, twist_update_trans, twist_update_rot, max_norm=1e3):

    if sum(torch.isnan(grad_Log_delta_R).flatten()) > 0:
        num_nans = sum(torch.isnan(grad_Log_delta_R).flatten())
        print(f"grad_Log_delta_R has {num_nans} nans out of {len(grad_Log_delta_R.flatten())}")
        # set rotation matrices to zero if they have nans
        grad_Log_delta_R = torch.nan_to_num(grad_Log_delta_R, nan=0.0)

    if sum(torch.isnan(trans_grad_t).flatten()) > 0:
        num_nans = sum(torch.isnan(trans_grad_t).flatten())
        print(f"trans_grad_t has {num_nans} nans out of {len(trans_grad_t.flatten())}")
        trans_grad_t = torch.nan_to_num(trans_grad_t, nan=0.0)

    # compute tangent vector update
    trans_update = trans_grad_t * d_t * trans_scale_t
    rotmats_update = so3_utils.hat(grad_Log_delta_R) * d_t * rot_scale_t

    norms = torch.norm(rotmats_update, dim=[-2, -1], keepdim=True) # keep the last dimensions
    # if sum(norms.flatten() > max_norm) > 0:
        # print("norms of rotmats_update are ", norms.shape)#, norms.flatten())
    grad_R_scaling = max_norm / (max_norm + norms)
    rotmats_update = grad_R_scaling*rotmats_update

    # apply Euler update
    if twist_update_trans:
        trans_t = trans_t + trans_update
    if twist_update_rot:
        rotmats_t = so3_utils.expmap(rotmats_t, skew_sym=rotmats_update)
    return trans_t, rotmats_t


def motif_rots_vec_F(trans_motif, R_motif, num_rots=1, align=True, scale=math.inf, device=torch.device('cpu'), dtype=torch.float64):
    """motif_offsets_and_rots_vec_F
    Args:
        L: number of residues in full scaffold
        motif_segments: list of tensors of shape [M_i, 7], where M_i is the number of residues in the ith motif segment
        motif_locations: list of list of tuples, each tuple is a (start, end) location for a segment,
            or None if we want to marginalize out motif_location.  When provided if there is only one
            set of motif locations, we use this set of locations for all items in the batch.  This is
            desirable in the case where we are doing particle filtering with the motif location fixed.
            Otherwise, motif_locations can be a list where each item in the list correspond to a example
            in the batch.
        num_rots: number of rotation matrices to include in conditioning

    Returns:
    F: a function that takes a tensor of shape [B, L, 7] and returns a tensor of shape [B, num_rots*num_offsets, M, 7]
    """
    # F_rots.shape = [num_rots, 3, 3]
    # Sample rotation uniformly
    if scale == math.inf:
        F_sampled_rots = so3_utils.sample_uniform(N=num_rots).to(device).to(dtype)
    else:
        F_sampled_rots = so3_utils.sample_wrapped_normal(N=num_rots, scale=scale).to(device).to(dtype)


    def F(R_pred, trans_pred):
        """F computes all projections of the prediction in a vectorized manner.
        Args:
            pred_tensor_7: tensor of shape [B, L, 7] representing the prediction
        """
        B = R_pred.shape[0]
        F_rots =  F_sampled_rots[None].expand((B, -1, -1, -1))  # [B, num_rots, 3, 3]
        R_pred, trans_pred = R_pred.to(dtype), trans_pred.to(dtype)

        # F_offsets_pred_trans = torch.einsum('OLM,BLi->BOMi', F_offsets, trans_pred)
        # F_offsets_pred_rots = torch.einsum('OLM,BLij->BOMij', F_offsets, R_pred)

        # Center segments of predictions of translations at [0., 0., 0.] by subtracting center of mass
        COMs = trans_pred.mean(dim=[-2], keepdim=True)
        trans_pred = trans_pred - COMs

        # with torch.no_grad():
        if align:  # find best alignement rotation
            pred_motif = all_atom.atom37_from_trans_rot(trans_pred, R_pred)   # [B, motif_res, 37, 3]
            motif = all_atom.atom37_from_trans_rot(trans_motif, R_motif)  # [1, motif_res, 37, 3]
            pred_motif = pred_motif[..., :3, :].reshape(B, -1, 3)  # [B, motif_res * 3, 3]
            motif = motif[..., :3, :].reshape(motif.shape[0], -1, 3)  # [1, motif_res * 3, 3]
            motif = motif.expand_as(pred_motif)  # [B, motif_res * 3, 3]
            _, _, align_rots = du.batch_align_structures(pred_motif, motif)  # [B, 3, 3]
            align_rots = align_rots[:, None].to(dtype)  # [B, 1, 3, 3]
            F_rots = torch.einsum('...ij,...jk->...ik', F_rots, align_rots)  # [B, num_rots, 3, 3]

        # Next apply rotations and reshape translations to [B, -1, M, 3], and rotations to [B, -1, M, 3, 3]
        # F_all_pred_trans = torch.einsum('Rij,BOMj->BORMi', F_rots, F_offsets_pred_trans)
        F_all_pred_trans = torch.einsum('BRij,BMj->BRMi', F_rots, trans_pred)
        # F_all_pred_trans = torch.reshape(F_all_pred_trans, [B, num_rots*O, M, 3])

        # Next apply rotations and reshape rotations to [B, -1, M, 3, 3]
        F_all_pred_rots = torch.einsum('BRij,BMjk->BRMik', F_rots, R_pred)
        # F_all_pred_rots = torch.reshape(F_all_pred_rots, [B, num_rots*O, M, 3, 3])

        return F_all_pred_rots, F_all_pred_trans

    return F


def get_all_motif_locations(L, segment_lengths, max_offsets=1000, first_call=True):
    """get_all_motif_locations returns of all possible starting and ending locations segments of length segment_lengths
    such that not of the segments overlap, the smallest starting location at least 0, and the largest ending location is at most L-1.

    The function works recursively.  First, it computes all possible starting and ending locations for the first segment.
    Then, it computes all possible starting and ending locations for the second segment, given the starting and ending locations

    Args:
        L: int, length of sequence
        segment_lengths: list of ints, length of each segment
        max_offsets: int, maximum number of motif offsets to return

    Returns:
        all_motif_locations: list of lists of tuples, each tuple is a (start, end) location for a segment
    """
    st_0_min = 0
    st_0_max = L - sum(segment_lengths)
    all_motif_locations = []
    for st_0 in range(st_0_min, st_0_max+1):
        end_0 = st_0 + segment_lengths[0] - 1

        # base case
        if len(segment_lengths) == 1:
            all_motif_locations.append([(st_0, end_0)])
        else:
            remaining_length = L - (end_0 + 1)
            all_later_motif_locs = get_all_motif_locations(
                remaining_length, segment_lengths[1:], max_offsets, first_call=False)
            for later_motif_locs in all_later_motif_locs:
                later_motif_locs  = [(st + end_0 + 1, end + end_0 + 1) for st, end in later_motif_locs]
                all_motif_locations.append(
                    [(st_0, end_0)] + later_motif_locs
                )

    if len(all_motif_locations) > max_offsets and first_call:
        # downsampling from len(all_motif_locations) to max_offsets offsets
        N = len(all_motif_locations)
        idcs = np.random.choice(N, max_offsets, replace=False)
        all_motif_locations = [all_motif_locations[idx] for idx in idcs]

    return all_motif_locations


def motif_offsets(L, motif_segments_length, motif_locations=None, max_offsets=1000, device=torch.device('cpu'), dtype=torch.float64):
    """motif_offsets returns a matrix F that pulls out the motif segments at the motif locations.

    """
    # If motif_locations is not None, then we are using a fixed motif location.
    # Set F to be a matrix that pulls out the motif segments at the fixed location
    if motif_locations is not None:
        # Set motif location to the one fixed location
        all_motif_locations = [motif_locations]
    else:
        # If motif_locations is None, then we are using a random motif location.
        segment_lengths = [segment_length for segment_length in motif_segments_length]
        all_motif_locations = get_all_motif_locations(L, segment_lengths, max_offsets)

    M = sum([segment_length for segment_length in motif_segments_length])
    F = torch.zeros([len(all_motif_locations), L, M], dtype=dtype, device=device)
    for i, motif_location in enumerate(all_motif_locations):
        motif_len_so_far = 0
        for segment_length, (st, end) in zip(motif_segments_length, motif_location):
            F[i, st:end+1, motif_len_so_far:motif_len_so_far+segment_length] = torch.eye(
                segment_length, dtype=dtype, device=device)
            motif_len_so_far += segment_length
    return F, all_motif_locations


def motif_offsets_and_rots_vec_F(L, motif_segments_length, motif_locations=None,
        num_rots=1, align=False, scale=math.inf, trans_motif=None, R_motif=None, max_offsets=1000, device=torch.device('cpu'),
        dtype=torch.float64, return_rots=True):
    """motif_offsets_and_rots_vec_F
    Args:
        L: number of residues in full scaffold
        motif_segments: list of tensors of shape [M_i, 7], where M_i is the number of residues in the ith motif segment
        motif_locations: list of list of tuples, each tuple is a (start, end) location for a segment,
            or None if we want to marginalize out motif_location.  When provided if there is only one
            set of motif locations, we use this set of locations for all items in the batch.  This is
            desirable in the case where we are doing particle filtering with the motif location fixed.
            Otherwise, motif_locations can be a list where each item in the list correspond to a example
            in the batch.
        num_rots: number of rotation matrices to include in conditioning

    Returns:
    F: a function that takes a tensor of shape [B, L, 7] and returns a tensor of shape [B, num_rots*num_offsets, M, 7]
    """
    M = sum([segment_length for segment_length in motif_segments_length])
    # if motif_locations is None or len(motif_locations) == 1:
    # print('motif_locations', motif_locations)
    F_offsets, all_motif_locations = motif_offsets(
        L, motif_segments_length, motif_locations=motif_locations, max_offsets=max_offsets,
        device=device, dtype=dtype)
    # print('all_motif_locations', all_motif_locations)
    O = len(all_motif_locations)

    # F_rots.shape = [num_rots, 3, 3]
    # F_rots = so3_utils.sample_uniform(N=num_rots).to(device).to(dtype)
    if scale == math.inf or scale >= 100:
        F_sampled_rots = so3_utils.sample_uniform(N=num_rots).to(device).to(dtype)
    else:
        F_sampled_rots = so3_utils.sample_wrapped_normal(N=num_rots, scale=scale).to(device).to(dtype)
    all_motif_locations_ = []
    for motif_location in all_motif_locations:
        all_motif_locations_.extend([motif_location for _ in range(num_rots)])
    motif_locations = all_motif_locations_

    def F(R_pred, trans_pred):
        """F computes all projections of the prediction in a vectorized manner.
        Args:
            pred_tensor_7: tensor of shape [B, L, 7] representing the prediction
        """
        B = R_pred.shape[0]
        # assert not align
        # F_rots = F_sampled_rots[None].expand((B, -1, -1, -1))  # [B, num_rots, 3, 3]
        F_rots = F_sampled_rots  # [num_rots, 3, 3]
        R_pred, trans_pred = R_pred.to(dtype), trans_pred.to(dtype)

        # First get and subset translations and translations
        F_offsets_pred_trans = torch.einsum('OLM,BLi->BOMi', F_offsets, trans_pred) # [B, O, M, 3]
        F_offsets_pred_rots = torch.einsum('OLM,BLij->BOMij', F_offsets, R_pred)

        # Center segments of predictions of translations at [0., 0., 0.] by subtracting center of mass
        COMs = F_offsets_pred_trans.mean(dim=[-2], keepdim=True)
        F_offsets_pred_trans = F_offsets_pred_trans - COMs

        # with torch.no_grad():
        if align:  # find best alignement rotation
            reshaped_F_offsets_pred_trans = rearrange(F_offsets_pred_trans, 'B O N C -> (B O) N C')
            reshaped_F_offsets_pred_rots = rearrange(F_offsets_pred_rots, 'B O N ... -> (B O) N ...')
            pred_motif = all_atom.atom37_from_trans_rot(reshaped_F_offsets_pred_trans, reshaped_F_offsets_pred_rots)   # [B*O, M, 37, 3]
            motif = all_atom.atom37_from_trans_rot(trans_motif, R_motif)  # [1, M, 37, 3]
            pred_motif = rearrange(pred_motif[..., :3, :], 'BO N K C -> BO (N K) C')
            motif = rearrange(motif[..., :3, :], 'B N K C -> B (N K) C')
            motif = motif.expand_as(pred_motif)  # [B, M * 3, 3]
            _, _, align_rots = du.batch_align_structures(pred_motif, motif)  # [BO, 3, 3]
            align_rots = align_rots.to(dtype)  # [BO, 3, 3]
            align_rots = rearrange(align_rots, '(B O) ... -> B O ...', O=O)  # [B, O, 3, 3]
            F_rots = torch.einsum('Rij,BOjk->BORik', F_rots, align_rots)  # [B, O, num_rots, 3, 3]
        else:
            F_rots = F_rots[None, None].expand((B, O, -1, -1, -1))

        # Next apply rotations and reshape translations to [B, -1, M, 3], and rotations to [B, -1, M, 3, 3]
        F_all_pred_trans = torch.einsum('BORij,BOMj->BORMi', F_rots, F_offsets_pred_trans)
        F_all_pred_trans = torch.reshape(F_all_pred_trans, [B, num_rots*O, M, 3])

        # Next apply rotations and reshape rotations to [B, -1, M, 3, 3]
        F_all_pred_rots = torch.einsum('BORij,BOMjk->BORMik', F_rots, F_offsets_pred_rots)
        F_all_pred_rots = torch.reshape(F_all_pred_rots, [B, num_rots*O, M, 3, 3])

        # # Next apply rotations and reshape translations to [B, -1, M, 3], and rotations to [B, -1, M, 3, 3]
        # F_all_pred_trans = torch.einsum('Rij,BOMj->BORMi', F_rots, F_offsets_pred_trans)
        # F_all_pred_trans = torch.reshape(F_all_pred_trans, [B, num_rots*O, M, 3])

        # # Next apply rotations and reshape rotations to [B, -1, M, 3, 3]
        # F_all_pred_rots = torch.einsum('Rij,BOMjk->BORMik', F_rots, F_offsets_pred_rots)
        # F_all_pred_rots = torch.reshape(F_all_pred_rots, [B, num_rots*O, M, 3, 3])

        return F_all_pred_rots, F_all_pred_trans

    if return_rots:
        return F, motif_locations, F_rots

    return F, motif_locations


def grad_log_lik_approx(
        R_pred, trans_pred, R_motif, trans_motif, Log_delta_R, delta_x, se3_diffuser, 
        rt_sq_trans, rt_sq_rot, F, twist_potential_rot=True, twist_potential_trans=True,
        ):
    """grad_log_lik_approx approximates gradients of conditional log likelihood
        grad_x log p(motif_x | X_t) and grad_R log p(motif_x | X_t)
        for p(motif_x | X_t) \propto \sum_{g\in F} N(motif_x; F(X), Var[x0|xt])

    Args:
        rigids_t, rigids_pred, rigids_motif: tensors of shape [B, L, 7], [B, L, 7] and [M, 7]
            xt, hat x0, and x0_M respectively.
    """
    logs = {}
    assert twist_potential_rot or twist_potential_trans

    # Compute vectorized version of F
    F_all_pred_rots, F_all_pred_trans = F(R_pred, trans_pred)
    
    trans_motif = trans_motif.to(F_all_pred_trans.dtype)
    R_motif = R_motif.to(F_all_pred_rots.dtype)

    log_p_rot_by_F, log_p_trans_by_F, rmsd = log_lik_approx(
        F_all_pred_rots, F_all_pred_trans, R_motif, trans_motif, se3_diffuser, rt_sq_trans, rt_sq_rot
        )
    log_p_by_F = 0.
    if twist_potential_rot:
        log_p_by_F += log_p_rot_by_F
    if twist_potential_trans:
        log_p_by_F += log_p_trans_by_F
    assert len(log_p_by_F.shape) == 2, log_p_by_F.shape # [B, |F|]

    log_p = log_p_by_F.logsumexp(dim=-1)
    log_p_rot = log_p_rot_by_F.logsumexp(dim=-1)
    log_p_trans = log_p_trans_by_F.logsumexp(dim=-1)
    # log_p = twist_scale * log_p

    # Find index with largest likelihood
    max_log_p_idx = log_p_by_F.argmax(dim=1)
    logs['log_p'] = log_p#.sum().item()
    logs['log_p_rot'] = log_p_rot#.sum().item()
    logs['log_p_trans'] = log_p_trans#.sum().item()
    logs['log_p_by_F_argmax'] = torch.tensor([log_p_by_F[i, idx] for i, idx in enumerate(max_log_p_idx)])
    logs['rmsd_argmax'] = torch.tensor([rmsd[i, idx] for i, idx in enumerate(max_log_p_idx)])
    

    # Compute gradients of log_p with respect to Log_delta_R and delta_x
    grad_Log_delta_R, grad_x = torch.autograd.grad(log_p.sum(), [Log_delta_R, delta_x])

    # Change the dtype of grad_x to match the dtype of rigids_t
    grad_x = grad_x.to(R_pred.dtype)
    grad_Log_delta_R = grad_Log_delta_R.to(R_pred.dtype)

    # skew_sym = so3_utils.hat(grad_Log_delta_R)
    # Compute Riemannian gradient for rotation
    # grad_R = torch.einsum('...ij,...jk->...ik', R_t, so3_utils.hat(grad_Log_delta_R))
    # return grad_R, grad_x, max_log_p_idx, log_p
    return grad_Log_delta_R, grad_x, logs, max_log_p_idx


def log_lik_approx(R_pred, x_pred, R_motif, trans_motif, se3_diffuser, rt_sq_trans, rt_sq_rot
):
    """log_lik_approx computes an approximation to p(motif_x | xt, f)

    The computation is done in a batched manner.

    Args:
        R_pred, x_pred: tensors of shapes [B, |F|, M, 3, 3] and  [B, |F|, M, 3]
        rigids_obs: tensors of shape [1, M, 7]

    Returns a tensor of shape [|F|] of log likelihoods
    """
    # Compute variance terms for likelihood approximations
    R_obs = R_motif
    x_obs = trans_motif

    # Compute term likelihood term for rotations
    # Frobenius norm approximation to tangent normal density
    # var_rot = ((1 - t_rot) / t_rot).squeeze()[:, None, None, None, None] ** 2 + obs_noise ** 2
    var_rot = rt_sq_rot[:, None, None]
    log_p_rot = -((R_obs - R_pred)**2 / 2 / var_rot).sum(dim=[-3, -2, -1])
    # log_p += -((R_pred - R_obs[None, None]).pow(2)/(
        # 4*sigma_t[:, None, None, None, None]**2)).sum(dim=[-3, -2, -1])

    # Compute term likelihood term for translations
    # log_p += -(((x_obs - x_pred)**2)/(2*(1-bar_a_t[:, None, None, None]))).sum(dim=[-1, -2])
    # var_trans = ((1 - t_trans) / t_trans).squeeze()[:, None, None, None] ** 2 + obs_noise ** 2
    var_trans = rt_sq_trans[:, None]
    log_p_trans = -((x_obs - x_pred)**2 / 2 / var_trans).sum(dim=[-1, -2])
    # print('log_p_trans', log_p_trans.shape)
    rmsd = ((x_obs - x_pred)**2).sum(-1).mean(-1).sqrt()
    # print('rmsd=', [f'{x:.2f}'for x in rmsd[0].tolist()])
    # print('argmin=', rmsd[0].argmin().item(), 'min=', rmsd[0][rmsd[0].argmin()].item())

    return log_p_rot, log_p_trans, rmsd
