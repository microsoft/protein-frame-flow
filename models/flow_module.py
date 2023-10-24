from typing import Any
import torch
import time
import os
import random
import wandb
import numpy as np
import pandas as pd
import logging
from pytorch_lightning import LightningModule
from analysis import metrics 
from analysis import utils as au
from models.flow_model import FlowModel
from models import utils as mu
from data.interpolant import Interpolant 
from data import utils as du
from data import all_atom
from data import so3_utils
from experiments import utils as eu
import shutil
import subprocess
from openfold.utils.superimposition import superimpose
from biotite.sequence.io import fasta
from pytorch_lightning.loggers.wandb import WandbLogger
import esm


class FlowModule(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._interpolant_cfg = cfg.interpolant

        # Set-up vector field prediction model
        self.model = FlowModel(cfg.model)

        # Set-up interpolant
        self.interpolant = Interpolant(cfg.interpolant)

        self._sample_write_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(self._sample_write_dir, exist_ok=True)

        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
        self.save_hyperparameters()

        self._folding_model = None
        
    def on_train_start(self):
        self._epoch_start_time = time.time()
        
    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'train/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )
        self._epoch_start_time = time.time()

    def model_step(self, noisy_batch: Any):
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch['res_mask'] * noisy_batch['diffuse_mask']
        if training_cfg.min_plddt_mask is not None:
            plddt_mask = noisy_batch['res_plddt'] > training_cfg.min_plddt_mask
            loss_mask *= plddt_mask
        num_batch, num_res = loss_mask.shape

        # Ground truth labels
        gt_trans_1 = noisy_batch['trans_1']
        gt_rotmats_1 = noisy_batch['rotmats_1']
        rotmats_t = noisy_batch['rotmats_t']
        gt_rot_vf = so3_utils.calc_rot_vf(
            rotmats_t, gt_rotmats_1.type(torch.float32))
        gt_bb_atoms = all_atom.to_atom37(gt_trans_1, gt_rotmats_1)[:, :, :3] 

        # Timestep used for normalization.
        r3_t = noisy_batch['r3_t']
        so3_t = noisy_batch['so3_t']
        r3_norm_scale = 1 - torch.min(
            r3_t[..., None], torch.tensor(training_cfg.t_normalize_clip))
        so3_norm_scale = 1 - torch.min(
            so3_t[..., None], torch.tensor(training_cfg.t_normalize_clip))
        
        # Model output predictions.
        model_output = self.model(noisy_batch)
        pred_trans_1 = model_output['pred_trans']
        pred_rotmats_1 = model_output['pred_rotmats']
        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)

        # Backbone atom loss
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3]
        gt_bb_atoms *= training_cfg.bb_atom_scale / r3_norm_scale[..., None]
        pred_bb_atoms *= training_cfg.bb_atom_scale / r3_norm_scale[..., None]
        loss_denom = torch.sum(loss_mask, dim=-1) * 3
        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2 * loss_mask[..., None, None],
            dim=(-1, -2, -3)
        ) / loss_denom

        # Translation VF loss
        trans_error = (gt_trans_1 - pred_trans_1) / r3_norm_scale * training_cfg.trans_scale
        trans_loss = training_cfg.translation_loss_weight * torch.sum(
            trans_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # Rotation VF loss
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / so3_norm_scale
        rots_vf_loss = training_cfg.rotation_loss_weights * torch.sum(
            rots_vf_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # Pairwise distance loss
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res*3, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res*3, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res*3])
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res*3])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask,
            dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) - num_res)

        se3_vf_loss = trans_loss + rots_vf_loss
        auxiliary_loss = (bb_atom_loss + dist_mat_loss) * (
            (r3_t[:, 0] > training_cfg.aux_loss_t_pass)
            & (so3_t[:, 0] > training_cfg.aux_loss_t_pass)
        )
        auxiliary_loss *= self._exp_cfg.training.aux_loss_weight
        se3_vf_loss += auxiliary_loss
        return {
            "bb_atom_loss": bb_atom_loss,
            "trans_loss": trans_loss,
            "dist_mat_loss": dist_mat_loss,
            "auxiliary_loss": auxiliary_loss,
            "rots_vf_loss": rots_vf_loss,
            "se3_vf_loss": se3_vf_loss
        }

    def validation_step(self, batch: Any, batch_idx: int):
        res_mask = batch['res_mask']
        self.interpolant.set_device(res_mask.device)
        num_batch, num_res = res_mask.shape
        samples = self.interpolant.sample(
            num_batch,
            num_res,
            self.model,
            trans_1=batch['trans_1'],
            rotmats_1=batch['rotmats_1']
        )[0][-1].numpy()

        batch_metrics = []
        for i in range(num_batch):

            # Write out sample to PDB file
            final_pos = samples[i]
            saved_path = au.write_prot_to_pdb(
                final_pos,
                os.path.join(
                    self._sample_write_dir,
                    f'sample_{i}_idx_{batch_idx}_len_{num_res}.pdb'),
                no_indexing=True
            )
            if isinstance(self.logger, WandbLogger):
                self.validation_epoch_samples.append(
                    [saved_path, self.global_step, wandb.Molecule(saved_path)]
                )
            try:
                mdtraj_metrics = metrics.calc_mdtraj_metrics(saved_path)
                ca_ca_metrics = metrics.calc_ca_ca_metrics(final_pos[:, metrics.CA_IDX])
                batch_metrics.append((mdtraj_metrics | ca_ca_metrics))
            except Exception:
                continue
        batch_metrics = pd.DataFrame(batch_metrics)
        self.validation_epoch_metrics.append(batch_metrics)
        
    def on_validation_epoch_end(self):
        if len(self.validation_epoch_samples) > 0:
            self.logger.log_table(
                key='valid/samples',
                columns=["sample_path", "global_step", "Protein"],
                data=self.validation_epoch_samples)
            self.validation_epoch_samples.clear()
        val_epoch_metrics = pd.concat(self.validation_epoch_metrics)
        for metric_name,metric_val in val_epoch_metrics.mean().to_dict().items():
            self._log_scalar(
                f'valid/{metric_name}',
                metric_val,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(val_epoch_metrics),
            )
        self.validation_epoch_metrics.clear()

    @torch.no_grad()
    def run_sde_sampling(
        self,
        batch: Any,
        return_traj,
        return_model_outputs,
        num_timesteps,
        vf_scale,
        sde_noise_scale,
        ):

        batch, _ = batch
        res_mask = batch['res_mask']
        device = res_mask.device
        num_batch, num_res = res_mask.shape[:2]
        trans_0 = self._centered_gaussian(
            (num_batch, num_res, 3), device) * du.NM_TO_ANG_SCALE
        rots_0 = torch.tensor(
            Rotation.random(num_batch*num_res).as_matrix(),
            device=device,
            dtype=torch.float32,
        ).view(num_batch, num_res, 3, 3)

        prot_traj = [(trans_0, rots_0)]
        if num_timesteps is None:
            num_timesteps = self._sampling_cfg.num_timesteps

        # Run sampling from t=1.0 back to t=0.0 (flipped definition of time)
        # Avoid singularities near both end points
        # use 1/num_timesteps because if d_t >> min_t then you can still get explosions
        ts = np.linspace(1.0 - (1/num_timesteps), self._exp_cfg.min_t, num_timesteps)
        t_1 = ts[0]
        model_traj = []
        for t_2 in ts[1:]:
            d_t = t_2 - t_1
            trans_t_1, rots_t_1 = prot_traj[-1]
            with torch.no_grad():
                if self._exp_cfg.noise_trans:
                    batch['trans_t'] = trans_t_1
                else:
                    batch['trans_t'] = batch['trans_1']
                if self._exp_cfg.noise_rots:
                    batch['rotmats_t'] = rots_t_1
                else:
                    batch['rotmats_t'] = batch['rotmats_1']

                # flip time
                batch['t'] = torch.ones((num_batch, 1)).to(device) * (1 - t_1)

                model_out = self.model(batch)
            pred_trans_1 = model_out['pred_trans']
            pred_rots_1 = model_out['pred_rots']
            pred_rotmats_1 = pred_rots_1.get_rot_mats()
            pred_rots_vf = model_out['pred_rots_vf']
            if self._exp_cfg.self_condition:
                batch['trans_sc'] = pred_trans_1

            model_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )
            assert d_t < 0

            # Euler-Maruyama step on the translations
            trans_score = ((1 - t_1) * pred_trans_1 - trans_t_1) / (t_1**2 * du.NM_TO_ANG_SCALE**2)
            trans_t_2 = (((-1)/(1-t_1)) * trans_t_1 - du.NM_TO_ANG_SCALE**2 * ((2 * t_1) / (1 - t_1)) * trans_score) * d_t + trans_t_1
            trans_t_2 = trans_t_2 + sde_noise_scale * torch.randn_like(trans_t_2) * np.sqrt( (-d_t) * (2 * t_1) / (1 - t_1)  ) * du.NM_TO_ANG_SCALE

            # ODE step on the rotations
            temp_scale = get_temp_scale(vf_scale, 1-t_1)

            if self._model_cfg.predict_rot_vf:
                rots_t_2 = so3_utils.geodesic_t(
                    # self._sampling_cfg.vf_scaling 
                    temp_scale * (-d_t) / t_1, pred_rotmats_1, rots_t_1, rot_vf=pred_rots_vf)
            else:
                rots_t_2 = so3_utils.geodesic_t(
                    # self._sampling_cfg.vf_scaling
                     temp_scale * (-d_t) / t_1, pred_rotmats_1, rots_t_1) 

            t_1 = t_2
            if return_traj:
                prot_traj.append((trans_t_2, rots_t_2))
            else:
                prot_traj[-1] = (trans_t_2, rots_t_2)

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rots_t_1 = prot_traj[-1]
        with torch.no_grad():
            if self._exp_cfg.noise_trans:
                batch['trans_t'] = trans_t_1
            else:
                batch['trans_t'] = batch['trans_1']
            if self._exp_cfg.noise_rots:
                batch['rotmats_t'] = rots_t_1
            else:
                batch['rotmats_t'] = batch['rotmats_1']
            batch['t'] = torch.ones((num_batch, 1)).to(device) * (1 - t_1)
            model_out = self.model(batch)

        pred_trans_1 = model_out['pred_trans']
        pred_rots_1 = model_out['pred_rots']
        pred_rotmats_1 = pred_rots_1.get_rot_mats()
        pred_rots_vf = model_out['pred_rots_vf'] # magnitude of this is tiny

        if self._model_cfg.predict_rot_vf:
            rots_t_2 = so3_utils.geodesic_t(1, None, rots_t_1, rot_vf=pred_rots_vf)
        else:
            rots_t_2 = pred_rotmats_1

        trans_t_2 = pred_trans_1

        model_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
        )

        if return_traj:
            prot_traj.append((trans_t_2, rots_t_2))
        else:
            prot_traj[-1] = (trans_t_2, rots_t_2)


        atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        if return_model_outputs:
            model_atom37_traj = all_atom.transrot_to_atom37(model_traj, res_mask)
            return atom37_traj, model_atom37_traj, model_traj
        return atom37_traj

    def _log_scalar(
            self,
            key,
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=None,
            sync_dist=False,
            rank_zero_only=True
        ):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )

    def training_step(self, batch: Any, stage: int):
        step_start_time = time.time()
        self.interpolant.set_device(batch['res_mask'].device)
        noisy_batch = self.interpolant.corrupt_batch(batch)
        if self._interpolant_cfg.self_condition and random.random() > 0.5:
            with torch.no_grad():
                model_sc = self.model(noisy_batch)
                noisy_batch['trans_sc'] = (
                    model_sc['pred_trans'] * noisy_batch['diffuse_mask'][..., None]
                    + noisy_batch['trans_1'] * (1 - noisy_batch['diffuse_mask'][..., None])
                )
        batch_losses = self.model_step(noisy_batch)
        num_batch = batch_losses['bb_atom_loss'].shape[0]
        total_losses = {
            k: torch.mean(v) for k,v in batch_losses.items()
        }
        for k,v in total_losses.items():
            self._log_scalar(
                f"train/{k}", v, prog_bar=False, batch_size=num_batch)
        
        # Losses to track. Stratified across t.
        so3_t = torch.squeeze(noisy_batch['so3_t'])
        self._log_scalar(
            "train/so3_t",
            np.mean(du.to_numpy(so3_t)),
            prog_bar=False, batch_size=num_batch)
        r3_t = torch.squeeze(noisy_batch['r3_t'])
        self._log_scalar(
            "train/r3_t",
            np.mean(du.to_numpy(r3_t)),
            prog_bar=False, batch_size=num_batch)
        for loss_name, loss_dict in batch_losses.items():
            if loss_name == 'rots_vf_loss':
                batch_t = so3_t
            else:
                batch_t = r3_t
            stratified_losses = mu.t_stratified_loss(
                batch_t, loss_dict, loss_name=loss_name)
            for k,v in stratified_losses.items():
                self._log_scalar(
                    f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # Training throughput
        self._log_scalar(
            "train/scaffolding_percent", 
            torch.mean(batch['diffuse_mask'].float()).item(), prog_bar=False, batch_size=num_batch)
        self._log_scalar(
            "train/length", batch['res_mask'].shape[1], prog_bar=False, batch_size=num_batch)
        self._log_scalar(
            "train/batch_size", num_batch, prog_bar=False)
        step_time = time.time() - step_start_time
        self._log_scalar(
            "train/examples_per_second", num_batch / step_time)
        train_loss = (
            total_losses[self._exp_cfg.training.loss]
            +  total_losses['auxiliary_loss']
        )
        self._log_scalar(
            "train/loss", train_loss, batch_size=num_batch)
        return train_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(),
            **self._exp_cfg.optimizer
        )

    def predict_step(self, batch, batch_idx):
        device = f'cuda:{torch.cuda.current_device()}'
        interpolant = Interpolant(self._infer_cfg.interpolant) 
        interpolant.set_device(device)

        sample_length = batch['num_res'].item()
        sample_id = batch['sample_id'].item()
        self._print_logger.info(f'Sampling instance {sample_id} length {sample_length}')
        atom37_traj, model_traj, _ = interpolant.sample(
            1, sample_length, self.model)
        
        sample_dir = os.path.join(
            self._output_dir, f'length_{sample_length}', f'sample_{sample_id}')
        top_sample_csv_path = os.path.join(sample_dir, 'top_sample.csv')
        if os.path.exists(top_sample_csv_path):
            self._print_logger.info(f'Skipping instance {sample_id} length {sample_length}')
            return
        os.makedirs(sample_dir, exist_ok=True)
        traj_paths = eu.save_traj(
            np.flip(du.to_numpy(torch.concat(atom37_traj, dim=0)), axis=0),
            np.flip(du.to_numpy(torch.concat(model_traj, dim=0)), axis=0),
            np.ones(sample_length),
            output_dir=sample_dir
        )

        # Run PMPNN to get sequences
        sc_output_dir = os.path.join(sample_dir, 'self_consistency')
        pdb_path = traj_paths['sample_path']
        os.makedirs(sc_output_dir, exist_ok=True)
        shutil.copy(pdb_path, os.path.join(
            sc_output_dir, os.path.basename(pdb_path)))
        os.makedirs(os.path.join(sc_output_dir, 'seqs'), exist_ok=True)

        self._print_logger.info(f'Sequence design instance {sample_id} length {sample_length}')
        output_path = os.path.join(sc_output_dir, "parsed_pdbs.jsonl")
        process = subprocess.Popen([
            'python',
            f'./ProteinMPNN/helper_scripts/parse_multiple_chains.py',
            f'--input_path={sc_output_dir}',
            f'--output_path={output_path}',
        ])
        _ = process.wait()
        pmpnn_args = [
            'python',
            f'./ProteinMPNN/protein_mpnn_run.py',
            '--out_folder',
            sc_output_dir,
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
            str(torch.cuda.current_device()),
        ]
        process = subprocess.Popen(
            pmpnn_args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
        _ = process.wait()
        mpnn_fasta_path = os.path.join(
            sc_output_dir,
            'seqs',
            os.path.basename(pdb_path).replace('.pdb', '.fa')
        )

        # Run ESMFold on each ProteinMPNN sequence and calculate metrics.
        self._print_logger.info(f'ESMFold instance {sample_id} length {sample_length}')
        mpnn_results = {
            'tm_score': [],
            'sample_pdb_path': [],
            'header': [],
            'sequence': [],
            'rmsd': [],
            'mean_plddt': [],
            'esmf_pdb_path': [],
        }
        if self._folding_model is None:
            device_idx = torch.cuda.current_device()
            device = f'cuda:{device_idx}'
            self._print_logger.info(f'Loading ESMFold on device {device}')
            torch.hub.set_dir(self._infer_cfg.pt_hub_dir)
            self._folding_model = esm.pretrained.esmfold_v1().eval()
            self._folding_model = self._folding_model.to(device)

        esmf_dir = os.path.join(sc_output_dir, 'esmf')
        os.makedirs(esmf_dir, exist_ok=True)
        fasta_seqs = fasta.FastaFile.read(mpnn_fasta_path)
        sample_feats = du.parse_pdb_feats('sample', pdb_path)
        for i, (header, string) in enumerate(fasta_seqs.items()):

            # Run ESMFold
            esmf_sample_path = os.path.join(esmf_dir, f'sample_{i}.pdb')

            with torch.no_grad():
                esmf_outputs = self._folding_model.infer(string)
                pdb_output = self._folding_model.output_to_pdb(esmf_outputs)[0]

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
            mpnn_results['rmsd'].append(rmsd)
            mpnn_results['tm_score'].append(tm_score)
            mpnn_results['esmf_pdb_path'].append(esmf_sample_path)
            mpnn_results['sample_pdb_path'].append(pdb_path)
            mpnn_results['header'].append(header)
            mpnn_results['sequence'].append(string)
            mpnn_results['mean_plddt'].append(mean_plddt)

        # Save results to CSV
        mpnn_results = pd.DataFrame(mpnn_results)
        mpnn_results.to_csv(os.path.join(sample_dir, 'sc_results.csv'))
        mpnn_results['length'] = sample_length
        mpnn_results['sample_id'] = sample_id
        del mpnn_results['header']
        del mpnn_results['sequence']

        # Select the top sample
        top_sample = mpnn_results.sort_values('rmsd', ascending=True).iloc[:1]

        # Compute secondary structure metrics
        sample_dict = top_sample.iloc[0].to_dict()
        ss_metrics = metrics.calc_mdtraj_metrics(sample_dict['sample_pdb_path'])
        top_sample['helix_percent'] = ss_metrics['helix_percent']
        top_sample['strand_percent'] = ss_metrics['strand_percent']
        top_sample.to_csv(top_sample_csv_path)
        self._print_logger.info(f'Done with sample {sample_id} length {sample_length}')
