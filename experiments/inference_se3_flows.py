"""Script for running inference and evaluation.

Assuming flow-matching is the base directory. Quick start command:
> python -W ignore experiments/inference_se3_flows.py

##########
# Config #
##########

Inference configs:
- configs/inference.yaml: Base config.

Most important fields:
- inference.num_gpus: Number of GPUs to use.
- inference.ckpt_path: Checkpoint to read model weights from.
- inference.write_to_wandb: Whether to write results to wandb.

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
│   │   │   ├── parsed_pdbs.jsonl       # ProteinMPNN compatible data file.
│   │   │   ├── sample.pdb              # Copy of sample_x/sample.pdb to use in ProteinMPNN
│   │   │   └── seqs                    # Directory of ProteinMPNN sequences.
│   │   │       └── sample.fa           # FASTA file of ProteinMPNN sequences.
│   │   ├── top_sample.csv              # CSV of the SC metrics for the best sequences and ESMFold structure.
│   │   ├── sc_results.csv              # All SC results from ProteinMPNN/ESMFold.
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

"""

import os
import time
import re
import numpy as np
import hydra
import torch
import pandas as pd
import glob
import wandb
import GPUtil
import shutil
import torch.distributed as dist
from pytorch_lightning import Trainer
from omegaconf import DictConfig, OmegaConf
from experiments import utils as eu
from models.flow_module import FlowModule
import subprocess

log = eu.get_pylogger(__name__)

WANDB_TABLE_COLS = [
    'tm_score', 'rmsd', 'helix_percent', 'strand_percent',
    'mean_plddt', 'length', 'sample_id', 
    'sample_pdb_path', 'esmf_pdb_path', 
    'esmf', 'sample'
]

FINAL_METRICS = [
    'scRMSD designable percent',
    'Helix percent',
    'Stand percent',
]


def _parse_designable(result_df):
    total_samples = result_df.shape[0]
    scrmsd_designable = (result_df.rmsd < 2.0).sum()
    return pd.DataFrame(data={ 
        'scRMSD designable': scrmsd_designable,
        'scRMSD designable percent': scrmsd_designable / total_samples, 
        'Helix percent': result_df.helix_percent.sum() / total_samples,
        'Stand percent': result_df.strand_percent.sum() / total_samples,
    }, index=[0])


def calc_novelty(novelty_path):
    foldseek_df = {
        'sample': [],
        'alntm': [],
    }
    with open(novelty_path) as file:
        for item in file:
            file, _, _, tm_score = item.split('\t')
            tm_score = float(tm_score)
            foldseek_df['sample'].append(file)
            foldseek_df['alntm'].append(tm_score)
    foldseek_df = pd.DataFrame(foldseek_df)
    novelty_summary = foldseek_df.groupby('sample').agg({'alntm': 'max'}).reset_index()
    return novelty_summary.alntm.mean()


class BlankDataset(torch.utils.data.Dataset):
    def __init__(self, samples_cfg):
        self._samples_cfg = samples_cfg
        all_sample_lengths = range(
            self._samples_cfg.min_length,
            self._samples_cfg.max_length+1,
            self._samples_cfg.length_step
        )
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


class Sampler:

    def __init__(self, cfg: DictConfig):
        """Initialize sampler.

        Args:
            cfg: inference config.
        """

        # Read in checkpoint.
        ckpt_path = cfg.inference.ckpt_path
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_cfg = OmegaConf.load(os.path.join(ckpt_dir, 'config.yaml'))

        # Set-up config.
        OmegaConf.set_struct(cfg, False)
        OmegaConf.set_struct(ckpt_cfg, False)
        cfg = OmegaConf.merge(cfg, ckpt_cfg)
        cfg.experiment.checkpointer.dirpath = './'
        self._cfg = cfg
        self._infer_cfg = cfg.inference
        self._samples_cfg = self._infer_cfg.samples
        self._rng = np.random.default_rng(self._infer_cfg.seed)

        # Set-up directories to write results to
        self._ckpt_name = '/'.join(ckpt_path.replace('.ckpt', '').split('/')[-3:])
        self._output_dir = os.path.join(
            self._infer_cfg.output_dir, self._ckpt_name, self._infer_cfg.name)
        os.makedirs(self._output_dir, exist_ok=True)
        log.info(f'Saving results to {self._output_dir}')
        config_path = os.path.join(self._output_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(config=self._cfg, f=f)
        log.info(f'Saving inference config to {config_path}')

        # Read checkpoint and initialize module.
        self._flow_module = FlowModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
        )
        self._flow_module.eval()
        self._flow_module._infer_cfg = self._infer_cfg
        self._flow_module._samples_cfg = self._samples_cfg
        self._flow_module._output_dir = self._output_dir

    def run_length_sampling(self):
        devices = GPUtil.getAvailable(
            order='memory', limit = 8)[:self._infer_cfg.num_gpus]
        log.info(f"Using devices: {devices}")

        blank_dataset = BlankDataset(self._samples_cfg)
        dataloader = torch.utils.data.DataLoader(
            blank_dataset, batch_size=1, shuffle=False, drop_last=False)
        trainer = Trainer(
            accelerator="gpu",
            strategy="ddp",
            devices=devices,
        )
        trainer.predict(
            self._flow_module, dataloaders=dataloader)
        
        all_csv_paths = glob.glob(self._output_dir+'/**/*.csv', recursive=True)
        top_sample_csv = pd.concat([
            pd.read_csv(x) for x in all_csv_paths if '/top_sample' in x
        ])
        top_sample_csv.to_csv(
            os.path.join(self._output_dir, 'all_top_samples.csv'), index=False)
        designable_csv = _parse_designable(top_sample_csv)
        designable_samples = top_sample_csv[top_sample_csv.rmsd <= 2.0]
        designable_dir = os.path.join(self._output_dir, 'designable')
        os.makedirs(designable_dir, exist_ok=True)
        designable_txt = os.path.join(designable_dir, 'designable.txt')
        if os.path.exists(designable_txt):
            os.remove(designable_txt)
        with open(designable_txt, 'w') as f:
            for _, row in designable_samples.iterrows():
                sample_path = row.sample_pdb_path
                sample_name = f'len_{row.length}_id_{row.sample_id}.pdb'
                write_path = os.path.join(designable_dir, sample_name)
                shutil.copy(sample_path, write_path)
                f.write(write_path+'\n')

        log.info(f'Running max cluster on {len(designable_samples)} samples')
        pmpnn_args = [
            './maxcluster64bit',
            '-l',
            designable_txt,
            os.path.join(designable_dir, 'all_by_all_lite'),
            '-C', '2',
            '-in',
            '-Rl',
            '-TM',
            '-Tm',
            '0.5',
        ]
        process = subprocess.Popen(
            pmpnn_args,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        stdout, _ = process.communicate()
        match = re.search(
            r"INFO\s*:\s*(\d+)\s+Clusters\s+@\s+Threshold\s+(\d+\.\d+)\s+\(\d+\.\d+\)",
            stdout.decode('utf-8'))
        clusters = int(match.group(1))
        designable_csv['Clusters'] = clusters
        designable_csv['Diversity'] = clusters / len(designable_samples)

        log.info(f'Running foldseek on {len(designable_samples)} samples')
        aln_path = os.path.join(designable_dir, 'aln_noise_01_seqs_100_esmf.m8')
        pmpnn_args = [
            'foldseek',
            'easy-search',
            designable_dir,
            '/Mounts/rbg-storage1/users/jyim/programs/foldseek/pdb',
            aln_path,
            'tmpFolder',
            '--alignment-type',
            '1',
            '--format-output',
            'query,target,alntmscore,lddt',
            '--tmscore-threshold',
            '0.0',
            '--exhaustive-search',
            '--max-seqs',
            '10000000000',
        ]
        process = subprocess.Popen(
            pmpnn_args,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        _ = process.wait()
        designable_csv['Novelty'] = calc_novelty(aln_path)
        designable_csv.to_csv(
            os.path.join(self._output_dir, 'designable.csv'), index=False)

    def write_to_wandb(self):
        top_sample_csv = pd.read_csv(
            os.path.join(self._output_dir, 'all_top_samples.csv'))
        designable_csv = pd.read_csv(
            os.path.join(self._output_dir, 'designable.csv'))

        cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
        wandb.init(
            config=dict(eu.flatten_dict(cfg_dict)),
            name=f'{self._ckpt_name}_{self._infer_cfg.name}',
            **self._infer_cfg.wandb
        )

        top_sample_csv = top_sample_csv.rename(
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
                wandb.Table(dataframe=top_sample_csv[['Sample length', 'scRMSD']]),
                x='Sample length',
                y='scRMSD',
                title='Length vs. scRMSD'
            )
        })

        top_sample_csv['designable'] = top_sample_csv.scRMSD < 2.0
        designable_summary = top_sample_csv[
            ['designable', 'Sample length']].groupby('Sample length').designable.mean().reset_index()
        wandb.log({
            "scRMSD passing" : wandb.plot.scatter(
                wandb.Table(dataframe=designable_summary),
                "Sample length",
                "designable"
            )
        })
        designable_summary.to_csv(
            os.path.join(self._output_dir, 'designable_per_length.csv'), index=False)

        top_sample_csv['esmf'] = top_sample_csv['esmf_pdb_path'].map(
            lambda x: wandb.Molecule(x))
        top_sample_csv['sample'] = top_sample_csv['sample_pdb_path'].map(
            lambda x: wandb.Molecule(x))
        wandb.log({
            "Top sample results": wandb.Table(
                dataframe=top_sample_csv
            )
        })
        data = [
            [label, val[0]]
            for (label, val) in designable_csv.to_dict().items()
            if label in FINAL_METRICS
        ]
        table = wandb.Table(data=data, columns = ["Metric", "value"])
        wandb.log({
            "Final metrics" : wandb.plot.bar(
                table, "Metric", "value", title="Final metrics")
        })
        strand_percent = top_sample_csv.groupby(
            'Sample length').Strand.mean().reset_index().round(2)
        helix_percent = top_sample_csv.groupby(
            'Sample length').Helix.mean().reset_index().round(2)
        wandb.log({
            "Strand composition": wandb.plot.scatter(
                wandb.Table(dataframe=strand_percent),
                x='Sample length',
                y='Strand',
                title='Strand composition'
            )
        })
        wandb.log({
            "Helix composition": wandb.plot.scatter(
                wandb.Table(dataframe=helix_percent),
                x='Sample length',
                y='Helix',
                title='Helix composition'
            )
        })


@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def run(cfg: DictConfig) -> None:

    # Read model checkpoint.
    log.info(f'Starting inference with {cfg.inference.num_gpus} GPUs')
    start_time = time.time()
    sampler = Sampler(cfg)
    sampler.run_length_sampling()
    if cfg.inference.write_to_wandb and dist.get_rank() == 0:
        dist.barrier()
        sampler.write_to_wandb()
    elapsed_time = time.time() - start_time
    log.info(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    run()