"""Script for running inference and evaluation.

Assuming flow-matching is the base directory. Quick start command:
> python -W ignore experiments/inference_se3_flows.py

##########
# Config #
##########

Inference configs:
- configs/inference.yaml: Base config.
- configs/inference_debug.yaml: used for debugging.
- configs/inference_worker.yaml: sampling without logging to wandb.

Most important fields:
- inference.ckpt_path: Checkpoint to read model weights from.
- inference_numtimesteps: Number of timesteps to use during sampling.
- inference.wandb_enable: Whether to turn on wandb logging.

#######################
# Directory structure #
#######################

inference_outputs/                      # Inference run name. Same as Wandb run.
├── config.yaml                         # Inference and model config.
├── length_N                            # Directory for samples of length N.
│   ├── sample_X                        # Directory for sample X of length N.
│   │   ├── bb_traj.pdb                 # Flow matching trajectory
│   │   ├── sample.pdb                  # Final sample (final step of trajectory).
│   │   ├── self_consistency            # Directory of SC intermediate files.
│   │   │   ├── esmf                    # Directory of ESMFold outputs.
│   │   │   │   └── sample_X.pdb        # ESMFold output.
│   │   │   ├── parsed_pdbs.jsonl       # ProteinMPNN compatible data file.
│   │   │   ├── sample.pdb              # Copy of sample_x/sample.pdb to use in ProteinMPNN
│   │   │   ├── sc_results.csv          # All SC results from ProteinMPNN/ESMFold.
│   │   │   └── seqs                    # Directory of ProteinMPNN sequences.
│   │   │       └── sample.fa           # FASTA file of ProteinMPNN sequences.
│   │   ├── top_sample.csv              # CSV of the SC metrics for the best sequences and ESMFold structure.
│   │   └── x0_traj.pdb                 # Model x0 trajectory.

###########
# Logging #
###########

By default inference_se3_flows.py will run inference and log results to
wandb. On wandb, we create scatter plots of designability results.

###############
## Workflow: ##
###############

Modify inference.yaml or use the command line to start a single GPU wandb logging run:

> python -W ignore experiments/inference_se3_flows.py inference.ckpt_path=<path>

Single GPU is too slow for 1000 timesteps. There is an option to just run sampling.
The following will not log to wandb.

> python -W ignore experiments/inference_se3_flows.py -cn inference_worker

After sampling is done, one can run inference with wandb logging.
This will pick up all the pre-computed samples and log complete metrics to wandb.

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
from biotite.sequence.io import fasta
import GPUtil
from typing import Optional

from analysis import utils as au
from analysis import metrics
from data import utils as du
from omegaconf import DictConfig, OmegaConf
from openfold.utils.superimposition import superimpose
import esm

from experiments import utils as eu
from models.flow_module import FlowModule
import wandb

log = eu.get_pylogger(__name__)

WANDB_TABLE_COLS = [
    'tm_score', 'rmsd', 'helix_percent', 'strand_percent',
    'mean_plddt', 'length', 'sample_id', 
    'sample_pdb_path', 'esmf_pdb_path', 
    'esmf', 'sample'
]


def _create_template_feats(num_res, device):
    return {        
        'res_mask': torch.ones(num_res, device=device)[None],
        'res_idx': torch.arange(1, num_res+1, device=device)[None],
    }


def _parse_designable(result_df):
    total_samples = result_df.shape[0]
    sctm_designable = (result_df.tm_score > 0.5).sum()
    sctm_confident = ((result_df.tm_score > 0.5) & (result_df.mean_plddt > 70)).sum()
    scrmsd_designable = (result_df.rmsd < 2.0).sum()
    scrmsd_confident = ((result_df.rmsd < 2.0) & (result_df.mean_plddt > 70)).sum()
    return pd.DataFrame(data={
        'scTM designable': sctm_designable,
        'scTM confident': sctm_confident,
        'scTM designable percent': sctm_designable / total_samples,
        'scTM confident percent': sctm_confident / total_samples, 

        'scRMSD designable': scrmsd_designable,
        'scRMSD confident': scrmsd_confident,
        'scRMSD designable percent': scrmsd_designable / total_samples, 
        'scRMSD confident percent': scrmsd_confident / total_samples, 

        'Helix percent': result_df.helix_percent.sum() / total_samples,
        'Stand percent': result_df.strand_percent.sum() / total_samples,
    }, index=[0])


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
            experiment_cfg=self._cfg.experiment,
            map_location=self.device
        ) 
        self._model.model = self._model.model.to(self.device)
        self._model.eval()
        self._model_ckpt = torch.load(ckpt_path, map_location=self.device)

        # Set-up directories to write results to
        output_base_dir = self._infer_cfg.output_dir
        ckpt_name = '_'.join(ckpt_path.replace('.ckpt', '').split('/')[-3:])
        mpnn_name = 'ca_mpnn' if cfg.inference.use_ca_pmpnn else 'frame_mpnn'
        interpolant_name = 'sde' if cfg.inference.do_sde else 'ode'
        vf_scale = 'temp_1' if cfg.inference.vf_scale is None else cfg.inference.vf_scale
        noise_scale = cfg.inference.sde_noise_scale
        if self._infer_cfg.name is None:
            self._output_dir = os.path.join(
                output_base_dir, ckpt_name,
                f'ts_{self._infer_cfg.num_timesteps}',
                mpnn_name,
                vf_scale,
                interpolant_name,
                f'NS_{noise_scale}'
            )
        else:
            self._output_dir = os.path.join(output_base_dir, self._infer_cfg.name) 
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

        # Set-up wandb
        if self._infer_cfg.name is None:
            wandb_name = f'{ckpt_name}_{interpolant_name}_ts_{self._infer_cfg.num_timesteps}_{vf_scale}_NS_{noise_scale}'
        else:
            wandb_name = self._infer_cfg.name
        if self._infer_cfg.wandb_enable:
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            wandb.init(
                config=dict(eu.flatten_dict(cfg_dict)),
                name=wandb_name,
                **self._infer_cfg.wandb
            )

    def _within_length_sampling(self, length_dir, sample_length):
        length_top_samples = []
        for sample_i in range(self._samples_cfg.samples_per_length):
            sample_dir = os.path.join(length_dir, f'sample_{sample_i}')
            sc_output_dir = os.path.join(sample_dir, 'self_consistency')
            top_sample_path = os.path.join(sample_dir, 'top_sample.csv')

            if not self._samples_cfg.overwrite:
                if self._infer_cfg.wandb_enable and os.path.exists(top_sample_path):
                    log.info(f'Skipping {sample_dir}')
                    top_sample = pd.read_csv(top_sample_path)
                    length_top_samples.append(top_sample)
                    continue
                elif (not self._infer_cfg.wandb_enable) and os.path.isdir(sample_dir):
                    log.info(f'Skipping {sample_dir}')
                    continue

            log.info(f'On sample {sample_i}')

            # Run sampling
            os.makedirs(sample_dir, exist_ok=True)
            init_feats = _create_template_feats(sample_length, self.device)
            atom37_traj, model_traj, _ = self._model.run_sampling(
                batch=(init_feats, 'sample'),
                return_traj=True,
                return_model_outputs=True,
                num_timesteps=self._infer_cfg.num_timesteps,
                do_sde=self._infer_cfg.do_sde,
                vf_scale=self._infer_cfg.vf_scale,
                sde_noise_scale=self._infer_cfg.sde_noise_scale,
            )
            traj_paths = self.save_traj(
                np.flip(du.to_numpy(torch.concat(atom37_traj, dim=0)), axis=0),
                np.flip(du.to_numpy(torch.concat(model_traj, dim=0)), axis=0),
                np.ones(sample_length),
                output_dir=sample_dir
            )

            # Run self-consistency
            pdb_path = traj_paths['sample_path']
            os.makedirs(sc_output_dir, exist_ok=True)
            shutil.copy(pdb_path, os.path.join(
                sc_output_dir, os.path.basename(pdb_path)))
            sc_results = self.run_self_consistency(
                sc_output_dir,
                pdb_path,
                motif_mask=None
            )
            sc_results['length'] = sample_length
            sc_results['sample_id'] = sample_i
            del sc_results['header']
            del sc_results['sequence']

            # Select the top sample
            top_sample = sc_results.sort_values('rmsd', ascending=True).iloc[:1]
            length_top_samples.append(top_sample)

            # Compute secondary structure metrics
            sample_dict = top_sample.iloc[0].to_dict()
            ss_metrics = metrics.calc_mdtraj_metrics(sample_dict['sample_pdb_path'])
            top_sample['helix_percent'] = ss_metrics['helix_percent']
            top_sample['strand_percent'] = ss_metrics['strand_percent']
            top_sample.to_csv(top_sample_path)
            log.info(f'Done sample {sample_i}')
        if len(length_top_samples):
            length_top_samples = pd.concat(length_top_samples)
        return length_top_samples

    def log_wandb_results(self, log_table, sc_summary):
        log_table = log_table.rename(
            columns={
                'length': 'Sample length',
                'rmsd': 'scRMSD',
                'tm_score': 'scTM',
                'helix_percent': 'Helix',
                'strand_percent': 'Strand'
            }
        ).round(2)
        wandb.log({
            "scRMSD": wandb.plot.scatter(
                wandb.Table(dataframe=log_table[['Sample length', 'scRMSD']]),
                x='Sample length',
                y='scRMSD',
                title='Length vs. scRMSD'
            )
        })
        wandb.log({
            "Secondary Structure": wandb.plot.scatter(
                wandb.Table(dataframe=log_table[['Helix', 'Strand']]),
                x='Helix',
                y='Strand',
                title='Secondary structure'
            )
        })

        log_table['esmf'] = log_table['esmf_pdb_path'].map(lambda x: wandb.Molecule(x))
        log_table['sample'] = log_table['sample_pdb_path'].map(lambda x: wandb.Molecule(x))
        wandb.log({
            "Top sample results": wandb.Table(
                dataframe=log_table
            )
        })

        designable_summary = pd.concat(sc_summary).rename(
            columns={
                'length': 'Sample length'
            }
        ).round(2)
        wandb.log({
            "scRMSD passing" : wandb.plot.scatter(
                wandb.Table(dataframe=designable_summary), "Sample length", "scTM designable")
        })

    def run_length_sampling(self):
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
        wandb_sc_summary = []
        wandb_table_rows = []
        for sample_length in all_sample_lengths:
            length_dir = os.path.join(
                self._output_dir, f'length_{sample_length}')
            os.makedirs(length_dir, exist_ok=True)
            log.info(f'Sampling length {sample_length}: {length_dir}')
            length_top_samples = self._within_length_sampling(
                length_dir, sample_length)
            if self._infer_cfg.wandb_enable:
                length_top_samples.to_csv(os.path.join(length_dir, 'top_samples.csv'))
                wandb_table_rows.append(length_top_samples)
                designable_results = _parse_designable(length_top_samples)
                designable_results['length'] = sample_length
                designable_results.to_csv(os.path.join(length_dir, 'sc_summary.csv'))
                wandb_sc_summary.append(designable_results)
                self.log_wandb_results(pd.concat(wandb_table_rows), wandb_sc_summary)
        if self._infer_cfg.wandb_enable:
            final_top_samples = pd.concat(wandb_table_rows)
            final_sc_summary = _parse_designable(final_top_samples).round(2)
            final_sc_summary.to_csv(os.path.join(self._output_dir, 'sc_summary.csv'))
            data = [[label, val[0]] for (label, val) in final_sc_summary.to_dict().items()]
            table = wandb.Table(data=data, columns = ["Metric", "value"])
            wandb.log({
                "Final metrics" : wandb.plot.bar(
                    table, "Metric", "value", title="Final metrics")
            })

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
            '--batch_size',
            '1',
            '--device',
            self.device.replace('cuda:', '')
        ]
        if self._infer_cfg.use_ca_pmpnn:
            pmpnn_args.append('--ca_only')

        os.makedirs(os.path.join(decoy_pdb_dir, 'seqs'), exist_ok=True)
        process = subprocess.Popen(pmpnn_args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        _ = process.wait()
        mpnn_fasta_path = os.path.join(
            decoy_pdb_dir,
            'seqs',
            os.path.basename(reference_pdb_path).replace('.pdb', '.fa')
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
            esmf_outputs = self.run_folding(string, esmf_sample_path)
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
            mpnn_results['sample_pdb_path'].append(reference_pdb_path)
            mpnn_results['header'].append(header)
            mpnn_results['sequence'].append(string)
            mpnn_results['mean_plddt'].append(mean_plddt)

        # Save results to CSV
        csv_path = os.path.join(decoy_pdb_dir, 'sc_results.csv')
        mpnn_results = pd.DataFrame(mpnn_results)
        mpnn_results.to_csv(csv_path)
        return mpnn_results

    def run_folding(self, sequence, save_path):
        """Run ESMFold on sequence."""
        with torch.no_grad():
            output = self._folding_model.infer(sequence)
            pdb_output = self._folding_model.output_to_pdb(output)[0]

        with open(save_path, "w") as f:
            f.write(pdb_output)
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
    sampler.run_length_sampling()
    elapsed_time = time.time() - start_time
    log.info(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    run()