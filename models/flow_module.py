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


def _to_atoms(trans, rots):
    num_batch, num_res, _ = trans.shape
    final_atom37 = all_atom.compute_backbone(
        du.create_rigid(rots, trans),
        torch.zeros(num_batch, num_res, 2, device=trans.device)
    )[0]
    return final_atom37


class FlowModule(LightningModule):

    def __init__(self, *, model_cfg, experiment_cfg):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = experiment_cfg
        self._sampling_cfg = experiment_cfg.sampling
        if model_cfg.architecture == 'flower':
            self.model = Flower(model_cfg.flower)
        elif model_cfg.architecture == 'framediff':
            self.model = VFModel(model_cfg.framediff)
        else:
            raise NotImplementedError()
        self._sample_write_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(self._sample_write_dir, exist_ok=True)
        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
        
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
    
    def _batch_ot(self, trans_1, trans_nm_0):
        batch_ot_cfg = self._exp_cfg.batch_ot
        num_batch, num_res = trans_1.shape[:2]

        num_noise = num_batch * batch_ot_cfg.noise_per_sample
        trans_nm_1 = trans_1 * du.ANG_TO_NM_SCALE

        noise_idx, gt_idx = torch.where(torch.ones(num_noise, num_batch))

        _, aligned_rmsd = superimposition.superimpose(trans_nm_0[noise_idx], trans_nm_1[gt_idx])
        cost_matrix = du.to_numpy(aligned_rmsd.reshape(num_noise, num_batch))
        noise_perm, gt_perm = linear_sum_assignment(cost_matrix)
        noise_perm = [x[1] for x in sorted(zip(gt_perm, noise_perm), key=lambda x: x[0])]
        trans_nm_0 = trans_nm_0[noise_perm]
        # TODO: We don't need to do alignment twice.
        aligned_trans_nm_0, _ = superimposition.superimpose(trans_nm_1, trans_nm_0)
        return aligned_trans_nm_0

    def _centered_gaussian(self, trans_reference):
        shape = trans_reference.shape
        noise = torch.randn(*shape, device=trans_reference.device)
        return noise - torch.mean(noise, dim=-2, keepdims=True)
    
    def _corrupt_batch(self, batch):
        gt_trans_1 = batch['trans_1']  # Angstrom
        gt_rotmats_1 = batch['rotmats_1']
        res_mask = batch['res_mask']
        device = gt_trans_1.device
        num_batch, num_res, _ = gt_trans_1.shape
        t = torch.rand(num_batch, 1, 1)
        batch['t'] = t[:, 0]

        trans_nm_0 = self._centered_gaussian(gt_trans_1)
        if self._exp_cfg.batch_ot.enabled:
            trans_nm_0 = self._batch_ot(gt_trans_1, trans_nm_0)

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
        if self._exp_cfg.training.superimpose not in ['all_atom', 'c_alpha', None]:
            raise ValueError(f'Unknown superimpose method {self._exp_cfg.training.superimpose}')
        gt_trans_1 = batch['trans_1']
        gt_rotmats_1 = batch['rotmats_1']
        gt_rotvecs_1 = so3_utils.rotmat_to_rotvec(gt_rotmats_1)
        res_mask = batch['res_mask']
        device = res_mask.device
        num_batch, num_res = res_mask.shape
        
        noisy_batch = self._corrupt_batch(batch)
        model_output = self.model(noisy_batch)
        pred_trans_1 = model_output['pred_trans']
        pred_rotmats_1 = model_output['pred_rotmats']
        pred_rotvecs_1 = so3_utils.rotmat_to_rotvec(pred_rotmats_1)

        gt_rot_vf = so3_utils.rot_vf(
            noisy_batch['rotmats_t'].type(torch.float32),
            gt_rotmats_1.type(torch.float32)
        )

        gt_bb_atoms = _to_atoms(gt_trans_1, gt_rotmats_1)[:, :, :4] 
        gt_bb_atoms = gt_bb_atoms.to(device) * res_mask[..., None, None]
        pred_bb_atoms = _to_atoms(pred_trans_1, pred_rotmats_1)[:, :, :4]
        pred_bb_atoms = pred_bb_atoms.to(device) * res_mask[..., None, None]

        if self._exp_cfg.training.superimpose == 'all_atom':
            flat_pred_bb_atoms = pred_bb_atoms.reshape(num_batch, num_res * 4, 3)
            flat_gt_bb_atoms = gt_bb_atoms.reshape(num_batch, num_res * 4, 3)
            gt_bb_atoms, _, super_rot, _ = superimposition.superimpose(
                flat_pred_bb_atoms, flat_gt_bb_atoms, return_transform=True)
            gt_bb_atoms = gt_bb_atoms.reshape(num_batch, num_res, 4, 3)
            gt_trans_1 = gt_bb_atoms[:, :, 1]
            gt_rotmats_1 = ru.rot_matmul(gt_rotmats_1, super_rot) 
            pred_rotvecs_1 = so3_utils.rotmat_to_rotvec(gt_rotmats_1)
            gt_rot_vf = so3_utils.rot_vf(
                noisy_batch['rotmats_t'].type(torch.float32),
                gt_rotmats_1.type(torch.float32)
            )
        elif self._exp_cfg.training.superimpose == 'c_alpha':
            gt_trans_1, _, super_rot, super_tran = superimposition.superimpose(
                pred_trans_1, gt_trans_1, return_transform=True)
            super_tran = super_tran.type(gt_bb_atoms.dtype)
            super_rot = super_rot.type(gt_bb_atoms.dtype)
            gt_bb_atoms = torch.einsum('bnai,bij->bnaj', gt_bb_atoms, super_rot) + super_tran[:, None, None, :]
            gt_rotmats_1 = ru.rot_matmul(gt_rotmats_1, super_rot) 
            pred_rotvecs_1 = so3_utils.rotmat_to_rotvec(gt_rotmats_1)
        else:
            raise ValueError(f'Unknown superimpose method {self._exp_cfg.training.superimpose}')
        
        gt_bb_atoms *= du.ANG_TO_NM_SCALE
        pred_bb_atoms *= du.ANG_TO_NM_SCALE
        gt_trans_1 *= du.ANG_TO_NM_SCALE
        pred_trans_1 *= du.ANG_TO_NM_SCALE
        
        num_res_per_batch = torch.sum(res_mask, dim=-1)

        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2 * res_mask[..., None, None],
            dim=(-1, -2, -3)
        ) / (num_res_per_batch * 3)

        trans_loss = torch.sum(
            (gt_trans_1 - pred_trans_1) ** 2 * res_mask[..., None],
            dim=(-1, -2)
        ) / (num_res_per_batch * 3)
        
        rots_loss = 0.5 * torch.sum(
            (gt_rotvecs_1 - pred_rotvecs_1) ** 2 * res_mask[..., None],
            dim=(-1, -2)
        ) / (num_res_per_batch * 3)
        
        rots_vf_loss = 0.5 * torch.sum(
            (gt_rot_vf - model_output['pred_rots_vf']) ** 2 * res_mask[..., None],
            dim=(-1, -2)
        ) / (num_res_per_batch * 3)

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
                    f'sample_{i}_len_{num_res}_epoch_{self.current_epoch}.pdb'),
                no_indexing=False
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
    def run_sampling(self, batch: Any, return_traj=False, return_model_outputs=False):
        batch, pdb_names = batch
        if self.current_epoch == 0:
            self._print_logger.info(f'Running eval on batches from {pdb_names}')
        res_mask = batch['res_mask']
        device = res_mask.device
        num_batch, num_res = res_mask.shape[:2]
        trans_0 = self._centered_gaussian((num_batch, num_res, 3)).to(device) * NM_TO_ANG_SCALE
        rots_0 = torch.tensor(
            Rotation.random(num_batch*num_res).as_matrix()
        ).float().reshape(num_batch, num_res, 3, 3).to(device)
        
        trans_traj = [trans_0]
        rots_traj = [rots_0]
        ts = np.linspace(self._sampling_cfg.min_t, 1.0, self._sampling_cfg.num_timesteps)
        t_1 = ts[0]
        model_outputs = []
        for t_2 in ts[1:]:
            d_t = t_2 - t_1
            trans_t_1 = trans_traj[-1]
            rots_t_1 = rots_traj[-1]
            with torch.no_grad():
                batch['trans_t'] = trans_t_1
                batch['rotmats_t'] = rots_t_1
                batch['t'] = torch.ones((num_batch, 1)).to(device) * t_1
                model_out = self.forward(batch)
                model_outputs.append(
                    tree.map_structure(lambda x: du.to_numpy(x), model_out)
                )

            pred_trans_1 = model_out['pred_trans']
            pred_rots_1 = model_out['pred_rotmats']

            trans_vf = (pred_trans_1 - trans_t_1) / (1 - t_1)
            trans_t_2 = trans_t_1 + trans_vf * d_t
            rots_t_2 = so3_utils.geodesic_t(d_t / (1 - t_1), pred_rots_1, rots_t_1)
            t_1 = t_2
            if return_traj:
                trans_traj.append(trans_t_2)
                rots_traj.append(rots_t_2)
            else:
                trans_traj[-1] = trans_t_2
                rots_traj[-1] = rots_t_2

        atom37_traj = []
        res_mask = res_mask.detach().cpu()
        for trans, rots in zip(trans_traj, rots_traj):
            rigids = du.create_rigid(rots, trans)
            atom37 = all_atom.compute_backbone(
                rigids,
                torch.zeros(
                    trans.shape[0],
                    trans.shape[1],
                    2
                )
            )[0]
            atom37 = atom37.detach().cpu()
            batch_atom37 = []
            for i in range(num_batch):
                batch_atom37.append(
                    du.adjust_oxygen_pos(atom37[i], res_mask[i])
                )
            atom37_traj.append(torch.stack(batch_atom37))

        if return_model_outputs:
            return atom37_traj, model_outputs
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
            "train/batch_size", num_batch, prog_bar=False)
        step_time = time.time() - step_start_time
        self._log_scalar("train/examples_per_second", num_batch / step_time)

        self._log_scalar(
            "train/loss", total_losses[self._exp_cfg.training.loss], batch_size=num_batch)
        return total_losses[self._exp_cfg.training.loss]

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
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
