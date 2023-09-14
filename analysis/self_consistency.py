import torch 
import os
from analysis import utils as au
import shutil
import subprocess
from biotite.sequence.io import fasta
from data import utils as du
import numpy as np
from analysis import metrics
from openfold.utils.superimposition import superimpose
import pandas as pd


def run_self_consistency(
    atom37,
    output_dir,
    batch_idx,
    folding_model,
    pmpnn_dir,
    seq_per_sample,
    use_ca_pmpnn,
):

    # expects atom37 tensor (batch_size, max_res, 37, 3)
    # output_dir where to save all the length folders
    # batch_idx integer 
    # folding_model the esm model on the correct device
    # pmpnn_dir directory for ProteinMPNN
    # seq_per_sample mpnn seqs per sample



    batch_size, max_res, _, _ = atom37.shape

    assert atom37.shape == (batch_size, max_res, 37, 3)


    for i in range(batch_size):
        sample_length = torch.count_nonzero(
            torch.sum(atom37[i, ...], dim=(1,2))
        ).item()
        length_dir = os.path.join(
            output_dir, f'length_{sample_length}')
        os.makedirs(length_dir, exist_ok=True)

        sample_dir = os.path.join(length_dir, f'sample_batch_{batch_idx}_device_{torch.cuda.current_device()}_idx_{i}')
        sc_output_dir = os.path.join(sample_dir, 'self_consistency')
        top_sample_path = os.path.join(sample_dir, 'top_sample.csv')
        os.makedirs(sample_dir, exist_ok=True)

        # cut it to the correct length here
        sample_np = atom37[i, 0:sample_length].cpu().numpy()
        sample_path = os.path.join(sample_dir, 'sample.pdb')
        au.write_prot_to_pdb(
            sample_np,
            sample_path,
            no_indexing=True
        )

        pdb_path = sample_path
        os.makedirs(sc_output_dir, exist_ok=True)
        shutil.copy(pdb_path, os.path.join(
            sc_output_dir, os.path.basename(pdb_path)))

        output_path = os.path.join(sc_output_dir, "parsed_pdbs.jsonl")
        process = subprocess.Popen([
            'python',
            f'{pmpnn_dir}/helper_scripts/parse_multiple_chains.py',
            f'--input_path={sc_output_dir}',
            f'--output_path={output_path}',
        ])
        _ = process.wait()
        pmpnn_args = [
            'python',
            f'{pmpnn_dir}/protein_mpnn_run.py',
            '--out_folder',
            sc_output_dir,
            '--jsonl_path',
            output_path,
            '--num_seq_per_target',
            str(seq_per_sample),
            '--sampling_temp',
            '0.1',
            '--seed',
            '38',
            '--batch_size',
            '1',
            '--device',
            str(torch.cuda.current_device()),
        ]
        if use_ca_pmpnn:
            pmpnn_args.append('--ca_only')

        os.makedirs(os.path.join(sc_output_dir, 'seqs'), exist_ok=True)
        process = subprocess.Popen(pmpnn_args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        _ = process.wait()
        mpnn_fasta_path = os.path.join(
            sc_output_dir,
            'seqs',
            os.path.basename(pdb_path).replace('.pdb', '.fa')
        )

        # Run ESMFold on each ProteinMPNN sequence and calculate metrics.
        mpnn_results = {
            'tm_score': [],
            'sample_pdb_path': [],
            'header': [],
            'sequence': [],
            'rmsd': [],
            'mean_plddt': [],
            'esmf_pdb_path': [],
        }
        motif_mask = None
        if motif_mask is not None:
            # Only calculate motif RMSD if mask is specified.
            mpnn_results['motif_rmsd'] = []

        esmf_dir = os.path.join(sc_output_dir, 'esmf')
        os.makedirs(esmf_dir, exist_ok=True)
        fasta_seqs = fasta.FastaFile.read(mpnn_fasta_path)
        sample_feats = du.parse_pdb_feats('sample', pdb_path)
        for i, (header, string) in enumerate(fasta_seqs.items()):

            # Run ESMFold
            esmf_sample_path = os.path.join(esmf_dir, f'sample_{i}.pdb')

            with torch.no_grad():
                esmf_outputs = folding_model.infer(string)
                pdb_output = folding_model.output_to_pdb(esmf_outputs)[0]

            with open(esmf_sample_path, "w") as f:
                f.write(pdb_output)

            mean_plddt = esmf_outputs['mean_plddt'][0].item()
            esmf_feats = du.parse_pdb_feats('folded_sample', esmf_sample_path)
            sample_seq = du.aatype_to_seq(sample_feats['aatype'])

            # Calculate scTM of ESMFold outputs with reference protein
            sample_bb_pos = sample_feats['bb_positions']
            esmf_bb_pos = esmf_feats['bb_positions']
            res_mask = np.ones(esmf_bb_pos.shape[0])
            _, tm_score = metrics.calc_tm_score(
                sample_bb_pos, esmf_bb_pos, sample_seq, sample_seq)
            _, rmsd = superimpose(
                torch.tensor(sample_bb_pos[None]),
                torch.tensor(esmf_bb_pos[None]),
                torch.tensor(res_mask[None])
            )
            rmsd = rmsd[0].item()
            if motif_mask is not None:
                sample_motif = sample_feats['bb_positions'][motif_mask]
                of_motif = esmf_feats['bb_positions'][motif_mask]
                motif_rmsd = metrics.calc_aligned_rmsd(
                    sample_motif, of_motif)
                mpnn_results['motif_rmsd'].append(motif_rmsd)
            mpnn_results['rmsd'].append(rmsd)
            mpnn_results['tm_score'].append(tm_score)
            mpnn_results['esmf_pdb_path'].append(esmf_sample_path)
            mpnn_results['sample_pdb_path'].append(pdb_path)
            mpnn_results['header'].append(header)
            mpnn_results['sequence'].append(string)
            mpnn_results['mean_plddt'].append(mean_plddt)

        # Save results to CSV
        csv_path = os.path.join(sc_output_dir, 'sc_results.csv')
        mpnn_results = pd.DataFrame(mpnn_results)
        mpnn_results.to_csv(csv_path)
