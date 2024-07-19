"""Script and utils for extracting motif segments into a list of tensor7 and
saving them out as pkls.

Sample command:
> python scripts/save_motif_segments.py
"""

import math
import pickle as pkl
import os
import time
import numpy as np
import torch
import pandas as pd
from datetime import datetime
import collections

from data import utils as du
from data import all_atom
from experiments import utils as eu
from openfold.utils import rigid_utils as ru
from openfold.data import data_transforms
from analysis import utils as au


def process_chain(design_pdb_feats):
    chain_feats = {
        'aatype': torch.tensor(design_pdb_feats['aatype']).long(),
        'all_atom_positions': torch.tensor(design_pdb_feats['atom_positions']).double(),
        'all_atom_mask': torch.tensor(design_pdb_feats['atom_mask']).double()
    }
    chain_feats = data_transforms.atom37_to_frames(chain_feats)
    chain_feats = data_transforms.make_atom14_masks(chain_feats)
    chain_feats = data_transforms.make_atom14_positions(chain_feats)
    chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)
    chain_feats['res_mask'] = design_pdb_feats['bb_mask']
    chain_feats['residue_index'] = design_pdb_feats['residue_index']
    return chain_feats

def create_pad_feats(pad_amt):
    pad_feats = {
        'rigids_impute': torch.zeros((pad_amt, 4, 4)),
    }
    return pad_feats

def process_motif_row(motif_row):
    """Parse row in the motif CSV."""
    motif_path = motif_row.motif_path
    motif_chain_feats = du.parse_pdb_feats(
        'motif', motif_path, chain_id=None)
    return {
        k: process_chain(v) for k,v in motif_chain_feats.items()
    }

def create_motif_feats(chain_feats, start_idx, end_idx):
    """Extract subset of features in chain_feats."""
    motif_length = end_idx - start_idx + 1
    motif_rigids = chain_feats['rigidgroups_gt_frames'][:, 0]
    pad_feats = {
        'rigids_impute': motif_rigids[start_idx:(end_idx+1)],
    }
    return pad_feats


def motif_locations_from_contig(sample_contig):
    # Parse contig.
    length_so_far = 0
    motif_locations = []
    for segment in sample_contig.split(','):
        if segment[0].isnumeric():
            length_so_far += int(segment.split('-')[0])
        else:
            lengths = segment[1:]
            start_idx, end_idx = lengths.split('-') # The end index is not inclusive
            len_motif_segment = int(end_idx) - int(start_idx) + 1
            motif_locations.append((length_so_far,
                length_so_far + len_motif_segment - 1))
            length_so_far += len_motif_segment
    return motif_locations

def process_contig(sample_contig, all_chain_feats):
    """process_contig extracts a list of rigids corresponding to the
    disjoint segments of a motif.

    Args:
        sample_contig: Contig to sample.
        all_chain_feats: Dict of motif features.

    Returns:
        List of Tensor7 representations of motif segments. For instance, for contig
        [5-5, A2-6, 2-2, B3-7] we return a list of two tensor7 arrays of
        shapes [5, 7] and [5, 7] corresponding to the backbone frames on
        chains A and B and the corresponding residue indices.
    """
    # Parse contig.
    motif_rigids = []
    length_so_far = 0
    motif_locations = []
    motif_aatypes = []
    motif_atom_positions = []
    for segment in sample_contig.split(','):
        if segment[0].isnumeric():
            length_so_far += int(segment.split('-')[0])
        else:
            chain_id = segment[0]
            lengths = segment[1:]
            start_idx, end_idx = lengths.split('-') # The end index is not inclusive
            len_motif_segment = int(end_idx) - int(start_idx) + 1
            motif_locations.append((length_so_far,
                length_so_far + len_motif_segment - 1))
            length_so_far += len_motif_segment
            chain_feats = all_chain_feats[chain_id]
            res_idx = chain_feats['residue_index']
            if np.all(int(start_idx) != res_idx) or np.all(int(end_idx) != res_idx):
                raise ValueError('Failed at finding motif residue index')
            start_idx = np.argmax(int(start_idx) == res_idx)
            end_idx = np.argmax(int(end_idx) == res_idx)
            segment_feats = create_motif_feats(
                chain_feats, start_idx, end_idx)
            segment_tensor7 = ru.Rigid.from_tensor_4x4(
                segment_feats['rigids_impute']).to_tensor_7()
            motif_rigids.append(segment_tensor7.cpu().numpy())
            motif_aatypes.append(chain_feats['aatype'][start_idx:(end_idx+1)])
            motif_atom_positions.append(chain_feats['all_atom_positions'][start_idx:(end_idx+1)])
    return motif_rigids, motif_locations, length_so_far, motif_aatypes, motif_atom_positions

def load_contig_test_case(row):
    motif_chain_feats = process_motif_row(row)
    motif_length = row.length
    motif_contig = row.contig

    if isinstance(motif_length, str):
        motif_length = [int(x) for x in motif_length.split('-')]
        if len(motif_length) == 1:
            motif_length.append(int(motif_length[0]) + 1)
    elif np.isnan(motif_length):
        motif_length = None
    else:
        raise ValueError(f'Unrecognized length: {motif_length}')

    # Run multiple samples for each motif
    sample_contig, _, _ = eu.get_sampled_mask(motif_contig, motif_length)

    # Create input features with sampled contig.
    motif_segments, motif_locations, total_length, motif_aatypes, motif_atom_positions = process_contig(sample_contig[0], motif_chain_feats)
    # make sure only one chain
    contig_test_case = {
        "motif_segments": motif_segments,
        "contig": motif_contig,
        "sampled_contig": sample_contig,
        "motif_locations": motif_locations,
        "total_length": total_length,
        "aatype": motif_aatypes,
        'motif_atom_positions': motif_atom_positions,
    }
    return contig_test_case

def load_contigs_by_test_case(inpaint_df):
    contigs_by_test_case = {}
    for _, row in inpaint_df.iterrows():
        name = str(row.target)
        contigs_by_test_case[name] = load_contig_test_case(row)
    return contigs_by_test_case

def save_motifs(csv_path, motif_segments_base_dir):
    """Sets up inference run on inpainting.

    Runs inference based on unconditional config.
    - samples_per_motif: number of samples per motif.
    - target_csv: CSV with information about each motif target.

    All outputs are written to
        {output_dir}/inpainting/{date_time}
    where {output_dir} is created at initialization.
    """
    inpaint_csv = pd.read_csv(csv_path, index_col=0)
    contigs_by_test_case = load_contigs_by_test_case(inpaint_csv)
    for name, motif_contig_info in contigs_by_test_case.items():
        motif_segments = motif_contig_info["motif_segments"]
        contig = motif_contig_info["contig"]
        sampled_contig = motif_contig_info["sampled_contig"]

        # Save pdb file with motif segments concatenated together
        segs_stacked = torch.tensor(np.concatenate(motif_segments))
        psis = torch.zeros_like(segs_stacked[:, :2])
        atom37_0 = all_atom.compute_backbone(
            ru.Rigid.from_tensor_7(segs_stacked), psis)[0]
        motif_pdb_fn = motif_segments_base_dir + name + "_motif.pdb"
        au.write_prot_to_pdb(
            atom37_0.numpy(),
            motif_pdb_fn,
            no_indexing=True,
            overwrite=True
        )
        print(name)
        seg_lengths = [int(v.shape[0]) for v in motif_segments]
        print(" ".join(str(l) for l in seg_lengths), "num orderings:", math.comb(100-sum(seg_lengths)+len(seg_lengths),
            len(seg_lengths)))
        fn = motif_segments_base_dir + name + "_motif_segments.pkl"
        with open(fn, 'wb') as f:
            pkl.dump(motif_segments, f)

def run():
    motif_segments_base_dir = "./motif_scaffolding//targets/"
    target_csv = "./motif_scaffolding//benchmark.csv"

    # Read model checkpoint.
    print('Starting inference')
    start_time = time.time()
    save_motifs(target_csv, motif_segments_base_dir)
    print(f'Finished in {time.time() - start_time}s')

if __name__ == '__main__':
    run()
