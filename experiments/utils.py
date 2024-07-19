"""Utility functions for experiments."""
import logging
import torch
import os
import random
import GPUtil
import numpy as np
import pandas as pd
from analysis import utils as au
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from motif_scaffolding import save_motif_segments
from openfold.utils import rigid_utils as ru


class LengthDataset(torch.utils.data.Dataset):
    def __init__(self, samples_cfg):
        self._samples_cfg = samples_cfg
        all_sample_lengths = range(
            self._samples_cfg.min_length,
            self._samples_cfg.max_length+1,
            self._samples_cfg.length_step
        )
        if samples_cfg.length_subset is not None:
            all_sample_lengths = [
                int(x) for x in samples_cfg.length_subset
            ]
        all_sample_ids = []
        for length in all_sample_lengths:
            for sample_id in range(self._samples_cfg.samples_per_length):
                all_sample_ids.append((length, sample_id))
        self._all_sample_ids = all_sample_ids

    def __len__(self):
        return len(self._all_sample_ids)

    def __getitem__(self, idx):
        num_res, sample_id = self._all_sample_ids[idx]
        batch = {
            'num_res': num_res,
            'sample_id': sample_id,
        }
        return batch


class ScaffoldingDataset(torch.utils.data.Dataset):
    def __init__(self, samples_cfg):
        self._samples_cfg = samples_cfg
        self._benchmark_df = pd.read_csv(self._samples_cfg.csv_path)
        if self._samples_cfg.target_subset is not None:
            self._benchmark_df = self._benchmark_df[
                self._benchmark_df.target.isin(self._samples_cfg.target_subset)
            ]
        if len(self._benchmark_df) == 0:
            raise ValueError('No targets found.')
        contigs_by_test_case = save_motif_segments.load_contigs_by_test_case(
            self._benchmark_df)

        num_batch = self._samples_cfg.num_batch
        assert self._samples_cfg.samples_per_target % num_batch == 0
        self.n_samples = self._samples_cfg.samples_per_target // num_batch

        all_sample_ids = []
        for row_id in range(len(contigs_by_test_case)):
            target_row = self._benchmark_df.iloc[row_id]
            for sample_id in range(self.n_samples):
                sample_ids = torch.tensor([num_batch * sample_id + i for i in range(num_batch)])
                all_sample_ids.append((target_row, sample_ids))
        self._all_sample_ids = all_sample_ids

    def __len__(self):
        return len(self._all_sample_ids)

    def __getitem__(self, idx):
        target_row, sample_id = self._all_sample_ids[idx]
        target = target_row.target
        motif_contig_info = save_motif_segments.load_contig_test_case(target_row)
        motif_segments = [
            torch.tensor(motif_segment, dtype=torch.float64)
            for motif_segment in motif_contig_info['motif_segments']]
        motif_locations  = []
        if isinstance(target_row.length, str):
            lengths = target_row.length.split('-')
            if len(lengths) == 1:
                start_length = lengths[0]
                end_length = lengths[0]
            else:
                start_length, end_length = lengths
            sample_lengths = [int(start_length), int(end_length)+1]
        else:
            sample_lengths = None
        sample_contig, sampled_mask_length, _ = get_sampled_mask(
            motif_contig_info['contig'], sample_lengths)
        motif_locations = save_motif_segments.motif_locations_from_contig(sample_contig[0])
        diffuse_mask = torch.ones(sampled_mask_length)
        trans_1 = torch.zeros(sampled_mask_length, 3)
        rotmats_1 = torch.eye(3)[None].repeat(sampled_mask_length, 1, 1)
        aatype = torch.zeros(sampled_mask_length)
        for (start, end), motif_pos, motif_aatype in zip(motif_locations, motif_segments, motif_contig_info['aatype']):
            diffuse_mask[start:end+1] = 0.0
            motif_rigid = ru.Rigid.from_tensor_7(motif_pos)
            motif_trans = motif_rigid.get_trans()
            motif_rotmats = motif_rigid.get_rots().get_rot_mats()
            trans_1[start:end+1] = motif_trans
            rotmats_1[start:end+1] = motif_rotmats
            aatype[start:end+1] = motif_aatype
        motif_com = torch.sum(trans_1, dim=-2, keepdim=True) / torch.sum(~diffuse_mask.bool())
        trans_1 = diffuse_mask[:, None] * trans_1 + (1 - diffuse_mask[:, None]) * (trans_1 - motif_com)
        return {
            'target': target,
            'sample_id': sample_id,
            'trans_1': trans_1,
            'rotmats_1': rotmats_1,
            'diffuse_mask': diffuse_mask,
            'aatype': aatype,
        }


def get_sampled_mask(contigs, length, rng=None, num_tries=1000000):
    '''
    Parses contig and length argument to sample scaffolds and motifs.

    Taken from rosettafold codebase.
    '''
    length_compatible=False
    count = 0
    while length_compatible is False:
        inpaint_chains=0
        contig_list = contigs.strip().split()
        sampled_mask = []
        sampled_mask_length = 0
        #allow receptor chain to be last in contig string
        if all([i[0].isalpha() for i in contig_list[-1].split(",")]):
            contig_list[-1] = f'{contig_list[-1]},0'
        for con in contig_list:
            if (all([i[0].isalpha() for i in con.split(",")[:-1]]) and con.split(",")[-1] == '0'):
                #receptor chain
                sampled_mask.append(con)
            else:
                inpaint_chains += 1
                #chain to be inpainted. These are the only chains that count towards the length of the contig
                subcons = con.split(",")
                subcon_out = []
                for subcon in subcons:
                    if subcon[0].isalpha():
                        subcon_out.append(subcon)
                        if '-' in subcon:
                            sampled_mask_length += (int(subcon.split("-")[1])-int(subcon.split("-")[0][1:])+1)
                        else:
                            sampled_mask_length += 1

                    else:
                        if '-' in subcon:
                            if rng is not None:
                                length_inpaint = rng.integers(int(subcon.split("-")[0]),int(subcon.split("-")[1]))
                            else:
                                length_inpaint=random.randint(int(subcon.split("-")[0]),int(subcon.split("-")[1]))
                            subcon_out.append(f'{length_inpaint}-{length_inpaint}')
                            sampled_mask_length += length_inpaint
                        elif subcon == '0':
                            subcon_out.append('0')
                        else:
                            length_inpaint=int(subcon)
                            subcon_out.append(f'{length_inpaint}-{length_inpaint}')
                            sampled_mask_length += int(subcon)
                sampled_mask.append(','.join(subcon_out))
        #check length is compatible 
        if length is not None:
            if sampled_mask_length >= length[0] and sampled_mask_length < length[1]:
                length_compatible = True
        else:
            length_compatible = True
        count+=1
        if count == num_tries: #contig string incompatible with this length
            raise ValueError("Contig string incompatible with --length range")
    return sampled_mask, sampled_mask_length, inpaint_chains


def dataset_creation(dataset_class, cfg, task):
    train_dataset = dataset_class(
        dataset_cfg=cfg,
        task=task,
        is_training=True,
    ) 
    eval_dataset = dataset_class(
        dataset_cfg=cfg,
        task=task,
        is_training=False,
    ) 
    return train_dataset, eval_dataset


def get_available_device(num_device):
    return GPUtil.getAvailable(order='memory', limit = 8)[:num_device]


def save_traj(
        sample: np.ndarray,
        bb_prot_traj: np.ndarray,
        x0_traj: np.ndarray,
        diffuse_mask: np.ndarray,
        output_dir: str,
        aatype = None,
    ):
    """Writes final sample and reverse diffusion trajectory.

    Args:
        bb_prot_traj: [T, N, 37, 3] atom37 sampled diffusion states.
            T is number of time steps. First time step is t=eps,
            i.e. bb_prot_traj[0] is the final sample after reverse diffusion.
            N is number of residues.
        x0_traj: [T, N, 3] x_0 predictions of C-alpha at each time step.
        aatype: [T, N, 21] amino acid probability vector trajectory.
        res_mask: [N] residue mask.
        diffuse_mask: [N] which residues are diffused.
        output_dir: where to save samples.

    Returns:
        Dictionary with paths to saved samples.
            'sample_path': PDB file of final state of reverse trajectory.
            'traj_path': PDB file os all intermediate diffused states.
            'x0_traj_path': PDB file of C-alpha x_0 predictions at each state.
        b_factors are set to 100 for diffused residues and 0 for motif
        residues if there are any.
    """

    # Write sample.
    diffuse_mask = diffuse_mask.astype(bool)
    sample_path = os.path.join(output_dir, 'sample.pdb')
    prot_traj_path = os.path.join(output_dir, 'bb_traj.pdb')
    x0_traj_path = os.path.join(output_dir, 'x0_traj.pdb')

    # Use b-factors to specify which residues are diffused.
    b_factors = np.tile((diffuse_mask * 100)[:, None], (1, 37))

    sample_path = au.write_prot_to_pdb(
        sample,
        sample_path,
        b_factors=b_factors,
        no_indexing=True,
        aatype=aatype,
    )
    prot_traj_path = au.write_prot_to_pdb(
        bb_prot_traj,
        prot_traj_path,
        b_factors=b_factors,
        no_indexing=True,
        aatype=aatype,
    )
    x0_traj_path = au.write_prot_to_pdb(
        x0_traj,
        x0_traj_path,
        b_factors=b_factors,
        no_indexing=True,
        aatype=aatype
    )
    return {
        'sample_path': sample_path,
        'traj_path': prot_traj_path,
        'x0_traj_path': x0_traj_path,
    }


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def flatten_dict(raw_dict):
    """Flattens a nested dict."""
    flattened = []
    for k, v in raw_dict.items():
        if isinstance(v, dict):
            flattened.extend([
                (f'{k}:{i}', j) for i, j in flatten_dict(v)
            ])
        else:
            flattened.append((k, v))
    return flattened
