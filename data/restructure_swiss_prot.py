"""Script for preprocessing Swiss-prot files.

python data/restructure_swiss_prot.py \
    --swiss_prot_dir /Mounts/rbg-storage1/users/jyim/large_data/swiss_prot \
    --write_dir /Mounts/rbg-storage1/users/jyim/large_data/swiss_prot_processed
"""

import argparse
import os
import multiprocessing as mp
import glob
import shutil


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
    default=50)

def process_fn(x):
    if not os.path.exists(x[2]):
        shutil.copy(x[1], x[2])
    print(f'Done with {x[0]}')

def main(args):
    write_dir = args.write_dir
    swiss_prot_dir = args.swiss_prot_dir

    # Set-up files to process
    all_swissprot_files = [
        x for x in glob.glob(swiss_prot_dir + '/*') if x.endswith('.cif.gz')]
    to_process = []
    prefix = 0
    for i, src_path in enumerate(all_swissprot_files):
        if (i % 1000) == 0:
            prefix += 1
            prefix_dir = os.path.join(write_dir, str(prefix))
            os.makedirs(prefix_dir, exist_ok=True)
        fname = os.path.basename(src_path)
        dest_path = os.path.join(prefix_dir, fname)
        to_process.append((i, src_path, dest_path))
    total_process_paths = len(to_process)
    print(f'{total_process_paths} files to process')

    with mp.Pool(processes=args.num_processes) as pool:
        _ = pool.map(process_fn, to_process)


if __name__ == "__main__":
    # Don't use GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = parser.parse_args()
    main(args)