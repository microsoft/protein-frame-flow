"""Script for running inference and evaluation.

Assuming flow-matching is the base directory. Quick start command:
> python -W ignore experiments/inference_se3_flows_ddp.py

##########
# Config #
##########

Inference configs:
- configs/inference_ddp.yaml: Base config for ddp inference

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
│   │   ├── bb_traj.pdb                 # TODO Currently not saved Flow matching trajectory
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
│   │   └── x0_traj.pdb                 # TODO Currently not saved Model x0 trajectory.

###########
# Logging #
###########

By default inference_se3_flows.py will run inference and log results to
wandb. On wandb, we create scatter plots of designability results.

###############
## Workflow: ##
###############

Change the number of GPUs to run by setting inference.num_gpus in the config

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
from glob import glob
import types

from analysis import utils as au
from analysis import metrics
from data import utils as du
from omegaconf import DictConfig, OmegaConf
from openfold.utils.superimposition import superimpose
import esm

from experiments import utils as eu
from models.flow_module import FlowModule
import wandb
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
import torch.distributed as dist

log = eu.get_pylogger(__name__)

WANDB_TABLE_COLS = [
    'tm_score', 'rmsd', 'helix_percent', 'strand_percent',
    'mean_plddt', 'length', 'sample_id', 
    'sample_pdb_path', 'esmf_pdb_path', 
    'esmf', 'sample'
]


class BlankDataset(torch.utils.data.Dataset):
    def __init__(self, min_res, max_res, num_samples_per_res_len):
        # Will sample num_samples_per_res_len samples for each res length between
        # min_res and max_res inclusive

        self.min_res = min_res
        self.max_res = max_res
        self.num_samples_per_res_len = num_samples_per_res_len

    def __len__(self):
        return self.num_samples_per_res_len * (self.max_res - self.min_res + 1)

    def __getitem__(self, idx):
        num_res = idx // self.num_samples_per_res_len + self.min_res
        res_mask = torch.zeros(self.max_res)
        res_mask[0:num_res] = 1.0
        template =  {        
            'res_mask': res_mask,
            'res_idx': torch.arange(1, self.max_res+1),
        }
        return template, 'sample'

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


def run_self_consistency(
    atom37: torch.Tensor, # (batch_size, max_res, 37, 3)
    output_dir: str, # where to save all the length folders
    batch_idx: int, # used for uniquely identifying samples
    folding_model, # the ESM model
    pmpnn_dir: str, # directory for ProteinMPNN
    seq_per_sample: int, # number of sequences to sample per sample
    use_ca_pmpnn: bool,
):

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


# This function will be added to the lightning model so that we can use
# trainer.predict to generate our samples with ddp
def predict_step(self, batch, batch_idx):

    # these class attributes need to be set manually before the trainer.predict call
    assert hasattr(self, '_infer_cfg')
    assert self._infer_cfg is not None
    assert hasattr(self, '_samples_cfg')
    assert hasattr(self, '_output_dir')
    assert hasattr(self, '_folding_model')

    atom37_traj = self.run_sampling(
        batch=batch,
        return_traj=False,
        return_model_outputs=False,
        num_timesteps=self._infer_cfg.num_timesteps,
        do_sde=self._infer_cfg.do_sde
    )
    # atom37_traj is list of length 1 with the element a tensor
    # of shape (batch_size, max_res, 37, 3)

    batch_size, max_res, _, _ = atom37_traj[0].shape
    assert atom37_traj[0].shape == (batch_size, max_res, 37, 3)
    assert len(atom37_traj) == 1

    assert batch[0]['res_mask'].shape == (batch_size, max_res)

    # zero out the masked residues for use in identifying length later
    atom37_traj[0] = atom37_traj[0] * batch[0]['res_mask'].view(batch_size, max_res, 1, 1).cpu()

    if self._folding_model is None:
        device_idx = torch.cuda.current_device()
        device = f'cuda:{device_idx}'
        torch.hub.set_dir(self._infer_cfg.pt_hub_dir)
        self._folding_model = esm.pretrained.esmfold_v1().eval()
        self._folding_model = self._folding_model.to(device)

    # save self consistency results to files
    run_self_consistency(
        atom37=atom37_traj[0],
        output_dir=self._output_dir,
        batch_idx=batch_idx,
        folding_model=self._folding_model,
        pmpnn_dir=self._infer_cfg.pmpnn_dir,
        seq_per_sample=self._samples_cfg.seq_per_sample,
        use_ca_pmpnn=self._infer_cfg.use_ca_pmpnn,
    )

    return atom37_traj


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

        # Set-up ckpt model
        self._model = FlowModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            model_cfg=self._cfg.model,
            experiment_cfg=self._cfg.experiment
        ) 
        self._model.eval()

        # Set-up directories to write results to
        output_base_dir = self._infer_cfg.output_dir
        self._ckpt_name = '_'.join(ckpt_path.replace('.ckpt', '').split('/')[-3:])
        if self._infer_cfg.name is None:
            self._output_dir = os.path.join(
                output_base_dir, self._ckpt_name, f'ts_{self._infer_cfg.num_timesteps}')
        else:
            self._output_dir = os.path.join(output_base_dir, self._infer_cfg.name) 
        os.makedirs(self._output_dir, exist_ok=True)
        log.info(f'Saving results to {self._output_dir}')
        config_path = os.path.join(self._output_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(config=self._cfg, f=f)
        log.info(f'Saving inference config to {config_path}')



    def ddp_sampling(self):

        batch_size_per_gpu = self._infer_cfg.batch_size_per_gpu
        num_gpus = self._infer_cfg.num_gpus
        samples_per_length = self._samples_cfg.samples_per_length

        blank_dataset = BlankDataset(min_res=self._samples_cfg.min_length, max_res=self._samples_cfg.max_length,
            num_samples_per_res_len=samples_per_length)

        dataloader = torch.utils.data.DataLoader(blank_dataset, batch_size=batch_size_per_gpu, shuffle=False)

        logger = CSVLogger(save_dir=self._output_dir, name="tmp") # Trainer requires some kind of logger

        trainer = Trainer(accelerator="gpu", strategy="ddp", devices=num_gpus,
                        logger=logger)

        self._model._infer_cfg = self._infer_cfg
        self._model._folding_model = None
        self._model._samples_cfg = self._samples_cfg
        self._model._output_dir = self._output_dir
        self._model.predict_step = types.MethodType(predict_step, self._model)

        trainer.predict(self._model, return_predictions=True, dataloaders=dataloader)

        dist.barrier()

        # finish up the analysis

        if dist.get_rank() != 0:
            return

        # Set-up wandb
        if self._infer_cfg.wandb_enable:
            log.info('Initializing wandb')
            cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
            wandb.init(
                config=dict(eu.flatten_dict(cfg_dict)),
                name=self._ckpt_name,
                **self._infer_cfg.wandb
            )

        wandb_sc_summary = []
        wandb_table_rows = []

        for sample_length in range(self._samples_cfg.min_length, self._samples_cfg.max_length+1):
            length_dir = os.path.join(
                self._output_dir, f'length_{sample_length}')
            os.makedirs(length_dir, exist_ok=True)
            length_top_samples = []

            sample_dirs = sorted(glob(os.path.join(length_dir, 'sample_batch_*')))

            for sample_dir in sample_dirs:
                sc_output_dir = os.path.join(sample_dir, 'self_consistency')
                top_sample_path = os.path.join(sample_dir, 'top_sample.csv')
                sc_result_path = os.path.join(sc_output_dir, 'sc_results.csv')

                sc_results = pd.read_csv(sc_result_path)

                sc_results['length'] = sample_length
                sc_results['sample_id'] = sample_dir.split('/')[-1]
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

            length_top_samples = pd.concat(length_top_samples)

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


@hydra.main(version_base=None, config_path="../configs", config_name="inference_ddp")
def run(cfg: DictConfig) -> None:

    # Read model checkpoint.
    log.info('Starting inference')
    start_time = time.time()
    sampler = Sampler(cfg)
    sampler.ddp_sampling()
    elapsed_time = time.time() - start_time
    log.info(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    run()