from typing import Any

import torch
import time
import os
import wandb
import numpy as np
import pandas as pd
import tree
import logging
from pytorch_lightning import LightningModule
from models.vf_model import VFModel
from models.genie_model import Genie
from models.flower_model import Flower
from data import all_atom 
from data import utils as du
from analysis import utils as au
from analysis import metrics
from scipy.spatial.transform import Rotation 
from openfold.utils import superimposition
from data import so3_utils
from pytorch_lightning.loggers.wandb import WandbLogger
from scipy.optimize import linear_sum_assignment
from openfold.utils import rigid_utils as ru

CA_IDX = metrics.CA_IDX


class FlowModule(LightningModule):

    def __init__(self, *, model_cfg, experiment_cfg):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = experiment_cfg
        self._sampling_cfg = experiment_cfg.sampling
        
        if model_cfg.architecture == 'genie':
            self._model_cfg = model_cfg.genie
            self.model = Genie(model_cfg.genie)
        elif model_cfg.architecture == 'framediff':
            self._model_cfg = model_cfg.framediff
            self.model = VFModel(model_cfg.framediff)
        elif model_cfg.architecture == 'flower':
            self._model_cfg = model_cfg.flower
            self.model = Flower(model_cfg.flower)
        else:
            raise NotImplementedError()
        self._sample_write_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(self._sample_write_dir, exist_ok=True)
        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
        self.save_hyperparameters()
        
    def on_train_start(self):
        self._epoch_start_time = time.time()
        
    def on_train_epoch_start(self):
        self.trainer.train_dataloader.batch_sampler.set_epoch(self.current_epoch)
        
    def on_train_epoch_end(self):
        epoch_time = time.time() - self._epoch_start_time
        self.log(
            'train/epoch_time_seconds',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )
        self._epoch_start_time = time.time()

    def forward(self, x):
        return self.model(x)
    
    def _batch_ot(self, trans_1, mask):
        batch_ot_cfg = self._exp_cfg.batch_ot
        num_batch, num_res = trans_1.shape[:2]
        device = trans_1.device
        num_noise = num_batch * batch_ot_cfg.noise_per_sample
        trans_nm_1 = trans_1 * du.ANG_TO_NM_SCALE

        # Sampose noise
        trans_nm_0 = self._centered_gaussian((num_noise, num_res, 3), device)

        # Align noise to ground truth
        if batch_ot_cfg.permute:
            noise_idx, gt_idx = torch.where(torch.ones(num_noise, num_batch))
            batch_nm_0 = trans_nm_0[noise_idx]
            batch_nm_1 = trans_nm_1[gt_idx]
            batch_mask = mask[gt_idx]
            aligned_nm_0, aligned_nm_1, _ = du.batch_align_structures(
                batch_nm_0, batch_nm_1, mask=batch_mask
            ) 
            aligned_nm_0 = aligned_nm_0.reshape(num_noise, num_batch, num_res, 3)
            aligned_nm_1 = aligned_nm_1.reshape(num_noise, num_batch, num_res, 3)
            
            # Compute cost matrix of aligned noise to ground truth
            batch_mask = batch_mask.reshape(num_noise, num_batch, num_res)
            cost_matrix = torch.sum(
                torch.linalg.norm(aligned_nm_0 - aligned_nm_1, dim=-1), dim=-1
            ) / torch.sum(batch_mask, dim=-1)
            noise_perm, gt_perm = linear_sum_assignment(du.to_numpy(cost_matrix))
            return aligned_nm_0[(tuple(gt_perm), tuple(noise_perm))]
        else:
            aligned_nm_0, _, _ = du.batch_align_structures(
                trans_nm_0, trans_nm_1, mask=mask
            )
            return aligned_nm_0

    def _centered_gaussian(self, shape, device):
        noise = torch.randn(*shape, device=device) * self._exp_cfg.sampling.prior_scale
        return noise - torch.mean(noise, dim=-2, keepdims=True)
    
    def _corrupt_batch(self, batch, t=None):
        gt_trans_1 = batch['trans_1']  # Angstrom
        gt_rotmats_1 = batch['rotmats_1']
        res_mask = batch['res_mask']
        device = gt_trans_1.device
        num_batch, num_res, _ = gt_trans_1.shape
        if t is None:
            if self._exp_cfg.training.t_sampler == 'bias':
                t_1 =  torch.rand(num_batch // 2, 1, 1, device=device) * 0.20
                t_2 =  torch.rand(num_batch - t_1.shape[0], 1, 1, device=device)
                t = torch.concat([t_1, t_2], dim=0)
            elif self._exp_cfg.training.t_sampler == 'uniform':
                t = torch.rand(num_batch, 1, 1, device=device)
            else:
                raise NotImplementedError()
        batch['t'] = t[:, 0]

        if self._exp_cfg.batch_ot.enabled:
            trans_nm_0 = self._batch_ot(gt_trans_1, res_mask)
        else:
            trans_nm_0 = self._centered_gaussian(gt_trans_1.shape, device) 

        if self._exp_cfg.noise_trans:
            trans_nm_t = (1 - t) * trans_nm_0 + t * gt_trans_1 * du.ANG_TO_NM_SCALE
            trans_nm_t *= res_mask[..., None]
            batch['trans_t'] = trans_nm_t * du.NM_TO_ANG_SCALE
        else:
            batch['trans_t'] = gt_trans_1

        if self._exp_cfg.noise_rots:
            rotmats_0 = torch.tensor(
                Rotation.random(num_res).as_matrix(),
                device=device,
                dtype=torch.float32
            )
            rotmats_t = so3_utils.geodesic_t(t, gt_rotmats_1, rotmats_0)
            rotmats_t = (
                rotmats_t * res_mask[..., None, None]
                + torch.eye(3, device=device)[None, None] * (1 - res_mask[..., None, None])
            )
            batch['rotmats_t'] = rotmats_t
        else:
            batch['rotmats_t'] = gt_rotmats_1
        return batch

    def model_step(self, batch: Any):
        training_cfg = self._exp_cfg.training
        if training_cfg.superimpose not in ['all_atom', 'c_alpha', None]:
            raise ValueError(f'Unknown superimpose method {training_cfg.superimpose}')
        gt_trans_1 = batch['trans_1']
        gt_rotmats_1 = batch['rotmats_1']
        res_mask = batch['res_mask']
        gt_bb_atoms = all_atom.to_atom37(gt_trans_1, gt_rotmats_1)[:, :, :4] 
        
        noisy_batch = self._corrupt_batch(batch)
        model_output = self.model(noisy_batch)
        pred_trans_1 = model_output['pred_trans']
        pred_rotmats_1 = model_output['pred_rotmats']
        pred_rots_vf = model_output['pred_rots_vf']
        pred_rotvecs_1 = so3_utils.rotmat_to_rotvec(gt_rotmats_1)

        gt_rotvecs_1 = so3_utils.rotmat_to_rotvec(gt_rotmats_1)
        gt_rot_vf = so3_utils.calc_rot_vf(
            noisy_batch['rotmats_t'].type(torch.float32),
            gt_rotmats_1.type(torch.float32)
        )

        pred_rotvecs_1 = so3_utils.rotmat_to_rotvec(pred_rotmats_1)
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :4]

        gt_bb_atoms *= du.ANG_TO_NM_SCALE
        pred_bb_atoms *= du.ANG_TO_NM_SCALE
        gt_trans_1 *= du.ANG_TO_NM_SCALE
        pred_trans_1 *= du.ANG_TO_NM_SCALE
        
        loss_denom = torch.sum(res_mask, dim=-1) * 3

        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2 * res_mask[..., None, None],
            dim=(-1, -2, -3)
        ) / loss_denom

        trans_loss = torch.sum(
            (gt_trans_1 - pred_trans_1) ** 2 * res_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom
        
        rots_loss = training_cfg.rotation_loss_weights * torch.sum(
            (gt_rotvecs_1 - pred_rotvecs_1) ** 2 * res_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom
        
        rots_vf_loss = training_cfg.rotation_loss_weights * torch.sum(
            (gt_rot_vf - pred_rots_vf) ** 2 * res_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        se3_loss = trans_loss + rots_loss
        se3_vf_loss = trans_loss + rots_vf_loss

        return noisy_batch, {
            "bb_atom_loss": bb_atom_loss,
            "trans_loss": trans_loss,
            "rots_loss": rots_loss,
            "se3_loss": se3_loss,
            "rots_vf_loss": rots_vf_loss,
            "se3_vf_loss": se3_vf_loss
        }

    def validation_step(self, batch: Any, _: int):
        samples = self.run_sampling(batch, return_traj=False)[0].numpy()
        num_batch, num_res = samples.shape[:2]
        
        batch_metrics = []
        for i in range(num_batch):

            # Write out sample to PDB file
            final_pos = samples[i]
            saved_path = au.write_prot_to_pdb(
                final_pos,
                os.path.join(
                    self._sample_write_dir,
                    f'sample_{i}_len_{num_res}.pdb'),
                no_indexing=True
            )
            if isinstance(self.logger, WandbLogger):
                self.validation_epoch_samples.append(
                    [saved_path, self.global_step, wandb.Molecule(saved_path)]
                )
            mdtraj_metrics = metrics.calc_mdtraj_metrics(saved_path)
            ca_ca_metrics = metrics.calc_ca_ca_metrics(final_pos[:, CA_IDX])
            batch_metrics.append((mdtraj_metrics | ca_ca_metrics))

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
        self._print_logger.info(f'Finished with eval epoch {self.current_epoch}')
        
    @torch.no_grad()
    def run_sampling(
            self,
            batch: Any,
            return_traj=False,
            return_model_outputs=False,
            num_timesteps=None,
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
        )
        if rots_0.ndim == 3:
            rots_0 = rots_0[None]
        
        prot_traj = [(trans_0, rots_0)]
        if num_timesteps is None:
            num_timesteps = self._sampling_cfg.num_timesteps
        ts = np.linspace(self._sampling_cfg.min_t, 1.0, num_timesteps)
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
                batch['t'] = torch.ones((num_batch, 1)).to(device) * t_1
                model_out = self.forward(batch)

            pred_trans_1 = model_out['pred_trans']
            pred_rots_1 = model_out['pred_rotmats']
            pred_rots_vf = model_out['pred_rots_vf']

            model_traj.append(
                (pred_trans_1.detach().cpu(), pred_rots_1.detach().cpu())
            )

            trans_vf = (pred_trans_1 - trans_t_1) / (1 - t_1)
            trans_t_2 = trans_t_1 + trans_vf * d_t
            if self._model_cfg.predict_rot_vf:
                rots_t_2 = so3_utils.geodesic_t(
                    d_t / (1 - t_1), pred_rots_1, rots_t_1, rot_vf=pred_rots_vf)
            else:
                rots_t_2 = so3_utils.geodesic_t(
                    d_t / (1 - t_1), pred_rots_1, rots_t_1) 
            t_1 = t_2
            if return_traj:
                prot_traj.append((trans_t_2, rots_t_2))
            else:
                prot_traj[-1] = (trans_t_2, rots_t_2)

        atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        if return_model_outputs:
            model_traj = all_atom.transrot_to_atom37(model_traj, res_mask)
            return atom37_traj, model_traj
        return atom37_traj
        
    def _log_scalar(self, key, value, on_step=True, on_epoch=False, prog_bar=True, batch_size=None, sync_dist=False, rank_zero_only=True):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key, value, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, batch_size=batch_size, sync_dist=sync_dist, rank_zero_only=rank_zero_only)

    def training_step(self, batch: Any, stage: int):
        step_start_time = time.time()
        _, batch_losses = self.model_step(batch)
        num_batch = batch_losses['bb_atom_loss'].shape[0]
        total_losses = {
            k: torch.mean(v) for k,v in batch_losses.items()
        }
        for k,v in total_losses.items():
            self._log_scalar(
                f"train/{k}", v, prog_bar=False, batch_size=num_batch)
        
        # Losses to track. Stratified across t.
        batch_t = torch.squeeze(batch['t'])
        self._log_scalar(
            "train/t",
            np.mean(du.to_numpy(batch_t)),
            prog_bar=False, batch_size=num_batch)
        for loss_name, loss_dict in batch_losses.items():
            stratified_losses = t_stratified_loss(
                batch_t, loss_dict, loss_name=loss_name)
            for k,v in stratified_losses.items():
                self._log_scalar(
                    f"train/{k}", v, prog_bar=False, batch_size=num_batch)
            
        # Training throughput
        self._log_scalar(
            "train/length", batch['res_mask'].shape[1], prog_bar=False, batch_size=num_batch)
        self._log_scalar(
            "train/batch_size", num_batch, prog_bar=False, rank_zero_only=False)
        step_time = time.time() - step_start_time
        self._log_scalar(
            "train/examples_per_second", num_batch / step_time, rank_zero_only=False)

        self._log_scalar(
            "train/loss", total_losses[self._exp_cfg.training.loss], batch_size=num_batch)
        return total_losses[self._exp_cfg.training.loss]

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.AdamW(
            params=self.model.parameters(),
            **self._exp_cfg.optimizer
        )
    
    
def t_stratified_loss(batch_t, batch_loss, num_bins=4, loss_name=None):
    """Stratify loss by binning t."""
    batch_t = du.to_numpy(batch_t)
    batch_loss = du.to_numpy(batch_loss)
    flat_losses = batch_loss.flatten()
    flat_t = batch_t.flatten()
    bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins+1)
    bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
    t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
    t_binned_n = np.bincount(bin_idx)
    stratified_losses = {}
    if loss_name is None:
        loss_name = 'loss'
    for t_bin in np.unique(bin_idx).tolist():
        bin_start = bin_edges[t_bin]
        bin_end = bin_edges[t_bin+1]
        t_range = f'{loss_name} t=[{bin_start:.2f},{bin_end:.2f})'
        range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
        stratified_losses[t_range] = range_loss
    return stratified_losses
