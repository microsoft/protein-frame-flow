"""Script for preprocessing Swiss-prot files.

python data/process_swiss_prot.py \
    --swiss_prot_dir /Mounts/rbg-storage1/users/jyim/large_data/swiss_prot_processed \
    --write_dir /Mounts/rbg-storage1/users/jyim/large_data/swiss_prot_pkls \
    --num_processes 100
"""

import argparse
import dataclasses
import functools as fn
import pandas as pd
import os
import multiprocessing as mp
import time
import glob
from Bio.PDB import PDBIO, MMCIFParser
import numpy as np
import mdtraj as md

from data import utils as du
from data import parsers
from data import errors
from data import mmcif_parsing
from experiments import utils as eu


# Define the parser
parser = argparse.ArgumentParser(
    description='PDB processing script.')
parser.add_argument(
    '--swiss_prot_dir',
    help='Path to directory with Swiss Prot files.',
    type=str)
parser.add_argument(
    '--write_dir',
    help='Path to write results to.',
    type=str)
parser.add_argument(
    '--num_processes',
    help='Number of processes.',
    type=int,
    default=100)
parser.add_argument(
    '--debug',
    help='Turn on for debugging.',
    action='store_true')
parser.add_argument(
    '--verbose',
    help='Whether to log everything.',
    action='store_true')



def process_file(input):
    """Processes protein file into usable, smaller pickles.

    Args:
        file_path: Path to file to read.
        write_dir: Directory to write pickles to.

    Returns:
        Saves processed protein to pickle and returns metadata.

    Raises:
        DataError if a known filtering rule is hit.
        All other errors are unexpected and are propogated.
    """
    src_path, dest_path = input
    metadata = {}
    prot_name = os.path.basename(src_path).replace('.cif', '')
    metadata['pdb_name'] = prot_name
    metadata['processed_path'] = dest_path
    metadata['raw_path'] = src_path

    try:
        with open(src_path, 'r') as f:
            parsed_mmcif = mmcif_parsing.parse(
                file_id=prot_name, mmcif_string=f.read())
    except:
        raise errors.FileExistsError(
            f'Error file do not exist {src_path}'
        )

    if parsed_mmcif.errors:
        raise errors.MmcifParsingError(
            f'Encountered errors {parsed_mmcif.errors}'
        )
    parsed_mmcif = parsed_mmcif.mmcif_object
    raw_mmcif = parsed_mmcif.raw_string
    if '_pdbx_struct_assembly.oligomeric_count' in raw_mmcif:
        raw_olig_count = raw_mmcif['_pdbx_struct_assembly.oligomeric_count']
        oligomeric_count = ','.join(raw_olig_count).lower()
    else:
        oligomeric_count = None
    if '_pdbx_struct_assembly.oligomeric_details' in raw_mmcif:
        raw_olig_detail = raw_mmcif['_pdbx_struct_assembly.oligomeric_details']
        oligomeric_detail = ','.join(raw_olig_detail).lower()
    else:
        oligomeric_detail = None
    metadata['oligomeric_count'] = oligomeric_count
    metadata['oligomeric_detail'] = oligomeric_detail

    # Extract all chains
    struct_chains = {
        chain.id.upper(): chain
        for chain in parsed_mmcif.structure.get_chains()}
    metadata['num_chains'] = len(struct_chains)

    # Extract features
    struct_feats = []
    all_seqs = set()
    for chain_id, chain in struct_chains.items():
        # Convert chain id into int
        chain_id = du.chain_str_to_int(chain_id)
        chain_prot = parsers.process_chain(chain, chain_id)
        chain_dict = dataclasses.asdict(chain_prot)
        chain_dict = du.parse_chain_feats(chain_dict)
        all_seqs.add(tuple(chain_dict['aatype']))
        struct_feats.append(chain_dict)

    complex_feats = du.concat_np_features(struct_feats, False)
    b_factor = complex_feats['b_factors']
    ca_plddt = b_factor[:, 1]
    metadata['avg_plddt'] = np.mean(ca_plddt)
    metadata['num_confident_plddt'] = np.mean(ca_plddt > 70)

    # Process geometry features
    complex_aatype = complex_feats['aatype']
    metadata['seq_len'] = len(complex_aatype)
    modeled_idx = np.where(complex_aatype != 20)[0]
    if np.sum(complex_aatype != 20) == 0:
        raise errors.LengthError('No modeled residues')
    min_modeled_idx = np.min(modeled_idx)
    max_modeled_idx = np.max(modeled_idx)
    metadata['modeled_seq_len'] = max_modeled_idx - min_modeled_idx + 1
    complex_feats['modeled_idx'] = modeled_idx
    
    try:
        
        # Workaround for MDtraj not supporting mmcif in their latest release.
        # MDtraj source does support mmcif https://github.com/mdtraj/mdtraj/issues/652
        # We temporarily save the mmcif as a pdb and delete it after running mdtraj.
        p = MMCIFParser()
        struc = p.get_structure("", src_path)
        io = PDBIO()
        io.set_structure(struc)
        pdb_path = src_path.replace('.cif', '.pdb')
        io.save(pdb_path)

        # MDtraj
        traj = md.load(pdb_path)
        # SS calculation
        pdb_ss = md.compute_dssp(traj, simplified=True)
        # DG calculation
        pdb_dg = md.compute_rg(traj)
        os.remove(pdb_path)
    except Exception as e:
        os.remove(pdb_path)
        raise errors.DataError(f'Mdtraj failed with error {e}')

    chain_dict['ss'] = pdb_ss[0]
    metadata['coil_percent'] = np.sum(pdb_ss == 'C') / metadata['modeled_seq_len']
    metadata['helix_percent'] = np.sum(pdb_ss == 'H') / metadata['modeled_seq_len']
    metadata['strand_percent'] = np.sum(pdb_ss == 'E') / metadata['modeled_seq_len']

    # Radius of gyration
    metadata['radius_gyration'] = pdb_dg[0]
    
    # Write features to pickles.
    du.write_pkl(dest_path, complex_feats)

    # Return metadata
    return metadata


def process_serially(paths_to_process):
    all_metadata = []
    for i, (src_path, dest_path) in enumerate(paths_to_process):
        try:
            start_time = time.time()
            metadata = process_file(src_path, dest_path)
            elapsed_time = time.time() - start_time
            print(f'Finished {src_path} in {elapsed_time:2.2f}s')
            all_metadata.append(metadata)
        except errors.DataError as e:
            print(f'Failed {src_path}: {e}')
    return all_metadata


def process_fn(
        file_path,
        verbose=None
    ):
    try:
        start_time = time.time()
        metadata = process_file(file_path)
        elapsed_time = time.time() - start_time
        if verbose:
            print(f'Finished {file_path} in {elapsed_time:2.2f}s')
        return metadata
    except errors.DataError as e:
        if verbose:
            print(f'Failed {file_path}: {e}')


def main(args):
    write_dir = args.write_dir
    swiss_prot_dir = args.swiss_prot_dir

    # Get all files to process
    print(f'Reading files from {swiss_prot_dir}')
    all_subdir = sorted(glob.glob(swiss_prot_dir + '/*'))
    to_process = []
    for subdir in all_subdir:
        subdir_files = glob.glob(subdir + '/*')
        prefix = os.path.basename(subdir)
        write_subdir = os.path.join(write_dir, prefix)
        os.makedirs(write_subdir, exist_ok=True)
        for src_path in subdir_files:
            fname = os.path.basename(src_path).replace('.cif', '.pkl')
            dest_path = os.path.join(write_subdir, fname)
            to_process.append((src_path, dest_path))
    total_num_paths = len(to_process)
    print(f'Found {total_num_paths} files to process')

    if args.debug:
        metadata_file_name = 'metadata_debug.csv'
        to_process = to_process[:100]
    else:
        metadata_file_name = 'metadata.csv'
    metadata_path = os.path.join(write_dir, metadata_file_name)
    print(f'Files will be written to {write_dir}')

    # Process each mmcif file
    if args.num_processes == 1 or args.debug:
        all_metadata = process_serially(to_process)
    else:
        _process_fn = fn.partial(
            process_fn,
            verbose=args.verbose)
        with mp.Pool(processes=args.num_processes) as pool:
            all_metadata = pool.map(_process_fn, to_process)
        all_metadata = [x for x in all_metadata if x is not None]
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(metadata_path, index=False)
    succeeded = len(all_metadata)
    print(
        f'Finished processing {succeeded}/{total_num_paths} files')


if __name__ == "__main__":
    # Don't use GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = parser.parse_args()
    main(args)