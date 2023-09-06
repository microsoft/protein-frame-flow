"""Script for running inference and sampling.

Sample command:
> python experiments/inference_se3_flows.py

"""

import os
import time
import tree
import numpy as np
import hydra
import torch
import subprocess
import pandas as pd
import shutil
from datetime import datetime
from biotite.sequence.io import fasta
import GPUtil
from typing import Optional

from analysis import utils as au
from analysis import metrics
from data import utils as du
from data import residue_constants
from omegaconf import DictConfig, OmegaConf
from openfold.data import data_transforms
from openfold.utils.superimposition import superimpose
import esm

from experiments import utils as eu
from models.flow_module import FlowModule
import wandb

log = eu.get_pylogger(__name__)


CA_IDX = residue_constants.atom_order['CA']


def _create_template_feats(num_res, device):
    return {        
        'res_mask': torch.ones(num_res, device=device)[None],
        'res_idx': torch.arange(1, num_res+1, device=device)[None],
    }


class Sampler:

    def __init__(self, cfg: DictConfig):
        """Initialize sampler.

        Args:
            cfg: inference config.
        """

        # Read in checkpoint
        ckpt_path = cfg.inference.ckpt_path
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_cfg = OmegaConf.load(os.path.join(ckpt_dir, 'config.yaml'))

        # Set-up config
        OmegaConf.set_struct(cfg, False)
        OmegaConf.set_struct(ckpt_cfg, False)
        cfg = OmegaConf.merge(cfg, ckpt_cfg)
        cfg.experiment.checkpointer.dirpath = './'
        self._cfg = cfg
        self._infer_cfg = cfg.inference
        self._samples_cfg = self._infer_cfg.samples
        self._rng = np.random.default_rng(self._infer_cfg.seed)

        # Set-up wandb
        if self._infer_cfg.wandb_enable:
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            wandb.init(
                config=dict(eu.flatten_dict(cfg_dict)),
                **cfg.wandb
            )

        # Set-up accelerator
        if torch.cuda.is_available() and self._infer_cfg.use_gpu:
            available_gpus = ''.join(
                [str(x) for x in GPUtil.getAvailable(
                    order='memory', limit = 8)])
            self.device = f'cuda:{available_gpus[0]}'
        else:
            self.device = 'cpu'
        log.info(f'Using device: {self.device}')

        # Set-up ckpt model
        self._model = FlowModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            model_cfg=self._cfg.model,
            experiment_cfg=self._cfg.experiment
        ) 
        self._model.model = self._model.model.to(self.device)
        self._model.eval()

        # Set-up directories to write results to
        self._output_dir = self._infer_cfg.output_dir
        os.makedirs(self._output_dir, exist_ok=True)
        log.info(f'Saving results to {self._output_dir}')
        config_path = os.path.join(self._output_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(config=self._cfg, f=f)
        log.info(f'Saving inference config to {config_path}')

        # Load models and experiment
        torch.hub.set_dir(self._infer_cfg.pt_hub_dir)
        self._folding_model = esm.pretrained.esmfold_v1().eval()
        self._folding_model = self._folding_model.to(self.device)

    def run_sampling(self):
        """Sets up inference run.

        All outputs are written to 
            {output_dir}/{date_time}
        where {output_dir} is created at initialization.
        """
        all_sample_lengths = range(
            self._samples_cfg.min_length,
            self._samples_cfg.max_length+1,
            self._samples_cfg.length_step
        )
        for sample_length in all_sample_lengths:
            length_dir = os.path.join(
                self._output_dir, f'length_{sample_length}')
            os.makedirs(length_dir, exist_ok=True)
            log.info(f'Sampling length {sample_length}: {length_dir}')

            for sample_i in range(self._samples_cfg.samples_per_length):
                sample_dir = os.path.join(length_dir, f'sample_{sample_i}')
                if os.path.isdir(sample_dir):
                    continue
                os.makedirs(sample_dir, exist_ok=True)
                init_feats = _create_template_feats(sample_length, self.device)
                atom37_traj, model_traj = self._model.run_sampling(
                    batch=(init_feats, 'sample'),
                    return_traj=True,
                    return_model_outputs=True,
                    num_timesteps=self._infer_cfg.num_timesteps,
                )
                traj_paths = self.save_traj(
                    np.flip(du.to_numpy(torch.concat(atom37_traj, dim=0)), axis=0),
                    np.flip(du.to_numpy(torch.concat(model_traj, dim=0)), axis=0),
                    np.ones(sample_length),
                    output_dir=sample_dir
                )

                # Run ProteinMPNN
                pdb_path = traj_paths['sample_path']
                sc_output_dir = os.path.join(sample_dir, 'self_consistency')
                os.makedirs(sc_output_dir, exist_ok=True)
                shutil.copy(pdb_path, os.path.join(
                    sc_output_dir, os.path.basename(pdb_path)))
                _ = self.run_self_consistency(
                    sc_output_dir,
                    pdb_path,
                    motif_mask=None
                )
                log.info(f'Done sample {sample_i}: {pdb_path}')

    def save_traj(
            self,
            bb_prot_traj: np.ndarray,
            x0_traj: np.ndarray,
            diffuse_mask: np.ndarray,
            output_dir: str
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
            bb_prot_traj[0],
            sample_path,
            b_factors=b_factors,
            no_indexing=True
        )
        prot_traj_path = au.write_prot_to_pdb(
            bb_prot_traj,
            prot_traj_path,
            b_factors=b_factors,
            no_indexing=True
        )
        x0_traj_path = au.write_prot_to_pdb(
            x0_traj,
            x0_traj_path,
            b_factors=b_factors,
            no_indexing=True
        )
        return {
            'sample_path': sample_path,
            'traj_path': prot_traj_path,
            'x0_traj_path': x0_traj_path,
        }

    def run_self_consistency(
            self,
            decoy_pdb_dir: str,
            reference_pdb_path: str,
            motif_mask: Optional[np.ndarray]=None):
        """Run self-consistency on design proteins against reference protein.
        
        Args:
            decoy_pdb_dir: directory where designed protein files are stored.
            reference_pdb_path: path to reference protein file
            motif_mask: Optional mask of which residues are the motif.

        Returns:
            Writes ProteinMPNN outputs to decoy_pdb_dir/seqs
            Writes ESMFold outputs to decoy_pdb_dir/esmf
            Writes results in decoy_pdb_dir/sc_results.csv
        """

        # Run PorteinMPNN
        output_path = os.path.join(decoy_pdb_dir, "parsed_pdbs.jsonl")
        process = subprocess.Popen([
            'python',
            f'{self._infer_cfg.pmpnn_dir}/helper_scripts/parse_multiple_chains.py',
            f'--input_path={decoy_pdb_dir}',
            f'--output_path={output_path}',
        ])
        _ = process.wait()
        pmpnn_args = [
            'python',
            f'{self._infer_cfg.pmpnn_dir}/protein_mpnn_run.py',
            '--out_folder',
            decoy_pdb_dir,
            '--jsonl_path',
            output_path,
            '--num_seq_per_target',
            str(self._samples_cfg.seq_per_sample),
            '--sampling_temp',
            '0.1',
            '--seed',
            '38',
            '--ze',
            '1',
            '--device',
            self.device.replace('cuda:', '')
        ]
        os.makedirs(os.path.join(decoy_pdb_dir, 'seqs'))
        process = subprocess.Popen(pmpnn_args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        ret = process.wait()
        mpnn_fasta_path = os.path.join(
            decoy_pdb_dir,
            'seqs',
            os.path.basename(reference_pdb_path).replace('.pdb', '.fa')
        )
        # Run ESMFold on each ProteinMPNN sequence and calculate metrics.
        mpnn_results = {
            'tm_score': [],
            'sample_path': [],
            'header': [],
            'sequence': [],
            'rmsd': [],
        }
        if motif_mask is not None:
            # Only calculate motif RMSD if mask is specified.
            mpnn_results['motif_rmsd'] = []
        esmf_dir = os.path.join(decoy_pdb_dir, 'esmf')
        os.makedirs(esmf_dir, exist_ok=True)
        fasta_seqs = fasta.FastaFile.read(mpnn_fasta_path)
        sample_feats = du.parse_pdb_feats('sample', reference_pdb_path)
        for i, (header, string) in enumerate(fasta_seqs.items()):

            # Run ESMFold
            esmf_sample_path = os.path.join(esmf_dir, f'sample_{i}.pdb')
            _ = self.run_folding(string, esmf_sample_path)
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
            rmsd = rmsd[0]
            if motif_mask is not None:
                sample_motif = sample_feats['bb_positions'][motif_mask]
                of_motif = esmf_feats['bb_positions'][motif_mask]
                motif_rmsd = metrics.calc_aligned_rmsd(
                    sample_motif, of_motif)
                mpnn_results['motif_rmsd'].append(motif_rmsd)
            mpnn_results['rmsd'].append(rmsd)
            mpnn_results['tm_score'].append(tm_score)
            mpnn_results['sample_path'].append(esmf_sample_path)
            mpnn_results['header'].append(header)
            mpnn_results['sequence'].append(string)

        # Save results to CSV
        csv_path = os.path.join(decoy_pdb_dir, 'sc_results.csv')
        mpnn_results = pd.DataFrame(mpnn_results)
        mpnn_results.to_csv(csv_path)

    def run_folding(self, sequence, save_path):
        """Run ESMFold on sequence."""
        with torch.no_grad():
            output = self._folding_model.infer_pdb(sequence)

        with open(save_path, "w") as f:
            f.write(output)
        return output

    def sample(self, sample_length: int):
        """Sample based on length.

        Args:
            sample_length: length to sample

        Returns:
            Sample outputs. See train_se3_diffusion.inference_fn.
        """
        # Process motif features.
        res_mask = np.ones(sample_length)
        fixed_mask = np.zeros_like(res_mask)

        # Initialize data
        ref_sample = self.diffuser.sample_ref(
            n_samples=sample_length,
            as_tensor_7=True,
        )
        res_idx = torch.arange(1, sample_length+1)
        init_feats = {
            'res_mask': res_mask,
            'seq_idx': res_idx,
            'fixed_mask': fixed_mask,
            'torsion_angles_sin_cos': np.zeros((sample_length, 7, 2)),
            'sc_ca_t': np.zeros((sample_length, 3)),
            **ref_sample,
        }
        # Add batch dimension and move to GPU.
        init_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats)
        init_feats = tree.map_structure(
            lambda x: x[None].to(self.device), init_feats)

        # Run inference
        sample_out = self.exp.inference_fn(
            init_feats,
            num_t=self._diff_conf.num_t,
            min_t=self._diff_conf.min_t, 
            aux_traj=True,
            noise_scale=self._diff_conf.noise_scale,
        )
        return tree.map_structure(lambda x: x[:, 0], sample_out)

@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def run(cfg: DictConfig) -> None:

    # Read model checkpoint.
    log.info('Starting inference')
    start_time = time.time()
    sampler = Sampler(cfg)
    sampler.run_sampling()
    elapsed_time = time.time() - start_time
    log.info(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    run()