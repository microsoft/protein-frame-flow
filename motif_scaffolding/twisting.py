import torch
from openfold.utils.rigid_utils import Rigid, Rotation
from data import so3_utils
from analysis import utils as au
import numpy as np
import time

def perturbations_for_grad(sample_feats, se3_diffuser):
    """perturbations_for_grad turns xt composed with an identity perturbation
    from which gradients may be computed.

    Args:
        sample_feats: dict, contains sample features, including xt and Rt which are perturbed
        se3_diffuser: SE3Diffuser, used to scale and unscale xt
    """
    device = sample_feats['R_t'].device
    Rt = sample_feats['R_t']
    xt = sample_feats['trans_t']

    delta_x = torch.zeros_like(xt, requires_grad=True)
    xt = se3_diffuser._r3_diffuser._scale(xt)
    xt = xt + delta_x
    xt = se3_diffuser._r3_diffuser._unscale(xt)

    Log_delta_R = torch.zeros_like(xt, requires_grad=True)
    delta_R = torch.einsum('...ij,...jk->...ik', Rt, so3_utils.hat(Log_delta_R))
    Rt = so3_utils.expmap(Rt, delta_R)

    xt = xt.to(device)
    Rt = Rt.to(device)

    # update rigids_t to include perturbed Rt and xt
    rigids_t = Rigid(rots=Rotation(rot_mats=Rt), trans=xt).to_tensor_7()

    sample_feats['trans_t'] = xt
    sample_feats['R_t'] = Rt
    sample_feats['rigids_t'] = rigids_t

    return Log_delta_R, delta_x

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

def motif_offsets(L, motif_segments, motif_locations=None, max_offsets=1000, device=torch.device('cpu')):
    """motif_offsets returns a matrix F that pulls out the motif segments at the motif locations.

    """
    # If motif_locations is not None, then we are using a fixed motif location.
    # Set F to be a matrix that pulls out the motif segments at the fixed location
    if motif_locations is not None:
        # Set motif location to the one fixed location
        all_motif_locations = [motif_locations]
    else:
        # If motif_locations is None, then we are using a random motif location.
        segment_lengths = [motif_segment.shape[0] for motif_segment in motif_segments]
        all_motif_locations = get_all_motif_locations(L, segment_lengths, max_offsets)

    M = sum([motif_segment.shape[0] for motif_segment in motif_segments])
    F = torch.zeros([len(all_motif_locations), L, M], dtype=motif_segments[0].dtype, device=device)
    for i, motif_location in enumerate(all_motif_locations):
        motif_len_so_far = 0
        for motif_segment, (st, end) in zip(motif_segments, motif_location):
            segment_length = motif_segment.shape[0]
            F[i, st:end+1, motif_len_so_far:motif_len_so_far+segment_length] = torch.eye(
                segment_length, dtype=motif_segment.dtype, device=device)
            motif_len_so_far += segment_length
    return F, all_motif_locations

def motif_offsets_and_rots_vec_F(L, motif_segments, motif_locations=None,
        num_rots=1, max_offsets=1000, device=torch.device('cpu'),
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
    M = sum([motif_segment.shape[0] for motif_segment in motif_segments])
    if motif_locations is None or len(motif_locations) == 1:
        F_offsets, all_motif_locations = motif_offsets(
            L, motif_segments, motif_locations=None, max_offsets=max_offsets,
            device=device)
        O = len(all_motif_locations)

        # F_rots.shape = [num_rots, 3, 3]
        F_rots = so3_utils.sample_uniform(N=num_rots).to(device).to(dtype)
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
            R_pred, trans_pred = R_pred.to(dtype), trans_pred.to(dtype)

            # First get and subset translations and translations
            F_offsets_pred_trans = torch.einsum('OLM,BLi->BOMi', F_offsets, trans_pred)
            F_offsets_pred_rots = torch.einsum('OLM,BLij->BOMij', F_offsets, R_pred)

            # Center segments of predictions of translations at [0., 0., 0.] by subtracting center of mass
            COMs = F_offsets_pred_trans.mean(dim=[-2], keepdim=True)
            F_offsets_pred_trans = F_offsets_pred_trans - COMs

            # Next apply rotations and reshape translations to [B, -1, M, 3], and rotations to [B, -1, M, 3, 3]
            F_all_pred_trans = torch.einsum('Rij,BOMj->BORMi', F_rots, F_offsets_pred_trans)
            F_all_pred_trans = torch.reshape(F_all_pred_trans, [B, num_rots*O, M, 3])

            # Next apply rotations and reshape rotations to [B, -1, M, 3, 3]
            F_all_pred_rots = torch.einsum('Rij,BOMjk->BORMik', F_rots, F_offsets_pred_rots)
            F_all_pred_rots = torch.reshape(F_all_pred_rots, [B, num_rots*O, M, 3, 3])

            return F_all_pred_rots, F_all_pred_trans

    else:
        # In the case that a batch of motif locations is specified, we don't need to enumerate all possible motif locations
        def F(R_pred, trans_pred):
            B = R_pred.shape[0]
            assert B == len(motif_locations), f"Batch size {B} does not match number of motif locations {len(motif_locations)}"
            assert R_pred.shape[0] == trans_pred.shape[0]
            R_pred_motif_by_batch, trans_pred_motif_by_batch = [], []
            for i in range(B):
                motif_location_i = motif_locations[i]
                R_pred_motif_by_batch.append(torch.concat([
                    R_pred[i, st:end+1] for (st, end) in motif_location_i]))
                trans_pred_motif_by_batch.append(torch.concat([
                    trans_pred[i, st:end+1] for (st, end) in motif_location_i]))
            R_pred_motif = torch.stack([
                R_pred_motif_by_batch[i] for i in range(R_pred.shape[0])])
            trans_pred_motif = torch.stack([
                trans_pred_motif_by_batch[i] for i in range(trans_pred.shape[0])])


            assert R_pred_motif.shape[0] == B, R_pred_motif.shape
            assert len(R_pred_motif.shape) == 4, R_pred_motif.shape # [B, M, 3, 3]

            # Add an extra dimension corresponding to the number of degrees of freedom
            R_pred_motif = R_pred_motif.unsqueeze(1)
            trans_pred_motif = trans_pred_motif.unsqueeze(1)
            return R_pred_motif, trans_pred_motif
    if return_rots:
        return F, motif_locations, F_rots

    return F, motif_locations


def grad_log_lik_approx(R_t, R_pred, trans_pred, motif_tensor_7, Log_delta_R, delta_x, se3_diffuser, t,
        F, twist_scale=1., twist_potential_rot=True, twist_potential_trans=True,
        ):
    """grad_log_lik_approx approximates gradients of conditional log likelihood
        grad_x log p(motif_x | X_t) and grad_R log p(motif_x | X_t)
        for p(motif_x | X_t) \propto \sum_{g\in F} N(motif_x; F(X), Var[x0|xt])

    Args:
        rigids_t, rigids_pred, rigids_motif: tensors of shape [B, L, 7], [B, L, 7] and [M, 7]
            xt, hat x0, and x0_M respectively.
    """
    # Compute vectorized version of F
    F_all_pred_rots, F_all_pred_trans = F(R_pred, trans_pred)
    log_p_by_F = log_lik_approx(
        F_all_pred_rots, F_all_pred_trans, motif_tensor_7, se3_diffuser, t,
         twist_potential_rot=twist_potential_rot, twist_potential_trans=twist_potential_trans,
        )
    assert len(log_p_by_F.shape) == 2, log_p_by_F.shape # [B, |F|]

    log_p = log_p_by_F.logsumexp(dim=-1)
    log_p = twist_scale * log_p

    # Find index with largest likelihood
    max_log_p_idx = log_p_by_F.argmax(dim=-1)

    # Compute gradients of log_p with respect to Log_delta_R and delta_x
    grad_Log_delta_R, grad_x = torch.autograd.grad(log_p.sum(), [Log_delta_R, delta_x])

    # Change the dtype of grad_x to match the dtype of rigids_t
    grad_x = grad_x.to(R_t.dtype)
    grad_Log_delta_R = grad_Log_delta_R.to(R_t.dtype)

    # Compute Riemannian gradient for rotation
    grad_R = torch.einsum('...ij,...jk->...ik', R_t, so3_utils.hat(grad_Log_delta_R))
    return grad_R, grad_x, max_log_p_idx, log_p

def log_lik_approx(R_pred, x_pred, rigids_obs, se3_diffuser, t,
         twist_potential_rot=True, twist_potential_trans=True,
):
    """log_lik_approx computes an approximation to p(motif_x | xt, f)

    The computation is done in a batched manner.

    Args:
        R_pred, x_pred: tensors of shapes [B, |F|, M, 3, 3] and  [B, |F|, M, 3]
        rigids_obs: tensors of shape [1, M, 7]

    Returns a tensor of shape [|F|] of log likelihoods
    """
    assert twist_potential_rot or twist_potential_trans

    # Compute variance terms for likelihood approximations
    bar_a_t = torch.exp(-se3_diffuser._r3_diffuser.marginal_b_t(t))
    sigma_t = torch.tensor(
        se3_diffuser._so3_diffuser.sigma(t.cpu().numpy()),
        dtype=R_pred.dtype, device=R_pred.device)

    R_obs = Rigid.from_tensor_7(rigids_obs).get_rots().get_rot_mats().to(R_pred.dtype)
    x_obs = Rigid.from_tensor_7(rigids_obs).get_trans()

    # Compute term likelihood term for rotations
    log_p = 0.
    if twist_potential_rot:
        # Frobenius norm approximation to tangent normal density
        log_p += -((R_pred - R_obs[None, None]).pow(2)/(
            4*sigma_t[:, None, None, None, None]**2)).sum(dim=[-3, -2, -1])

    # Compute term likelihood term for translations
    if twist_potential_trans:
        # scale down x_pred and x_obs
        x_pred = se3_diffuser._r3_diffuser._scale(x_pred)
        x_obs = se3_diffuser._r3_diffuser._scale(x_obs)

        log_p += -(((x_obs - x_pred)**2)/(2*(1-bar_a_t[:, None, None, None]))).sum(dim=[-1, -2])

    return log_p
