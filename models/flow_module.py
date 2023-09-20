from typing import Any

import torch
import time
import os
import random
import wandb
import numpy as np
import copy
import pandas as pd
import tree
import logging
from pytorch_lightning import LightningModule
from models.vf_model import VFModel
from models.genie_model import Genie
from models.flower_model import Flower
from models.flow_model import FlowModel
from data import all_atom 
from data import utils as du
from analysis import utils as au
from analysis import metrics
from scipy.spatial.transform import Rotation 
from data import so3_utils
from data import flow_utils
from pytorch_lightning.loggers.wandb import WandbLogger
from scipy.optimize import linear_sum_assignment
from openfold.utils import rigid_utils as ru
from data import r3_diffuser

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
        elif model_cfg.architecture == 'flow':
            self._model_cfg = model_cfg.flow
            self.model = FlowModel(model_cfg.flow)            
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
        noisy_batch = copy.deepcopy(batch)
        gt_trans_1 = batch['trans_1']  # Angstrom
        gt_rotmats_1 = batch['rotmats_1']
        res_mask = batch['res_mask']
        device = gt_trans_1.device
        num_batch, num_res, _ = gt_trans_1.shape
        if t is None:
            t = torch.rand(num_batch, 1, 1, device=device)
        noisy_batch['t'] = t[:, 0]

        if self._exp_cfg.batch_ot.enabled:
            trans_nm_0 = self._batch_ot(gt_trans_1, res_mask)
        else:
            trans_nm_0 = self._centered_gaussian(gt_trans_1.shape, device)
        noisy_batch['trans_0'] = trans_nm_0 * du.NM_TO_ANG_SCALE

        if self._exp_cfg.noise_trans:
            trans_nm_t = (1 - t) * trans_nm_0 + t * gt_trans_1 * du.ANG_TO_NM_SCALE
            trans_nm_t *= res_mask[..., None]
            noisy_batch['trans_t'] = trans_nm_t * du.NM_TO_ANG_SCALE
        else:
            noisy_batch['trans_t'] = gt_trans_1

        if self._exp_cfg.noise_rots:
            rotmats_0 = torch.tensor(
                Rotation.random(num_batch*num_res).as_matrix(),
                device=device,
                dtype=torch.float32
            )
            rotmats_0 = rotmats_0.reshape(num_batch, num_res, 3, 3)
            noisy_batch['rotmats_0'] = rotmats_0
            rotmats_t = so3_utils.geodesic_t(t, gt_rotmats_1, rotmats_0)
            rotmats_t = (
                rotmats_t * res_mask[..., None, None]
                + torch.eye(3, device=device)[None, None] * (1 - res_mask[..., None, None])
            )
            noisy_batch['rotmats_t'] = rotmats_t
        else:
            noisy_batch['rotmats_t'] = gt_rotmats_1
        return noisy_batch

    def model_step(self, noisy_batch: Any):
        training_cfg = self._exp_cfg.training
        if training_cfg.superimpose not in ['all_atom', 'c_alpha', None]:
            raise ValueError(f'Unknown superimpose method {training_cfg.superimpose}')
        gt_trans_1 = noisy_batch['trans_1']
        gt_rotmats_1 = noisy_batch['rotmats_1']
        res_mask = noisy_batch['res_mask']
        num_batch, num_res = res_mask.shape[:2]
        gt_bb_atoms = all_atom.to_atom37(gt_trans_1, gt_rotmats_1)[:, :, :3] 
        
        model_output = self.model(noisy_batch)
        pred_trans_1 = model_output['pred_trans']
        pred_rots_1 = model_output['pred_rots']
        pred_rotvecs_1 = pred_rots_1.get_rotvec()
        pred_rotmats_1 = pred_rots_1.get_rot_mats()
        pred_rots_vf = model_output['pred_rots_vf']

        gt_rotvecs_1 = so3_utils.rotmat_to_rotvec(gt_rotmats_1)
        gt_rot_vf = so3_utils.calc_rot_vf(
            noisy_batch['rotmats_t'].type(torch.float32),
            gt_rotmats_1.type(torch.float32)
        )

        pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3]
        gt_bb_atoms *= training_cfg.bb_atom_scale
        pred_bb_atoms *= training_cfg.bb_atom_scale

        loss_denom = torch.sum(res_mask, dim=-1) * 3

        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2 * res_mask[..., None, None],
            dim=(-1, -2, -3)
        ) / loss_denom

        t_norm_scale = 1 - (1 - self._exp_cfg.min_sigma)*noisy_batch['t'][..., None]
        gt_trans = gt_trans_1 * training_cfg.trans_scale
        gt_trans /= t_norm_scale 
        pred_trans = pred_trans_1 * training_cfg.trans_scale
        pred_trans /= t_norm_scale
        trans_loss = torch.sum(
            (gt_trans - pred_trans) ** 2 * res_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom
        
        rots_loss = training_cfg.rotation_loss_weights * torch.sum(
            (gt_rotvecs_1 - pred_rotvecs_1) ** 2 * res_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        rots_vf_error = (gt_rot_vf - pred_rots_vf) / t_norm_scale
        rots_vf_loss = training_cfg.rotation_loss_weights * torch.sum(
            rots_vf_error ** 2 * res_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # Pairwise distance loss
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res*3, 3]) * du.NM_TO_ANG_SCALE
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res*3, 3]) * du.NM_TO_ANG_SCALE
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(res_mask[:, :, None], (1, 1, 3))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res*3])
        flat_res_mask = torch.tile(res_mask[:, :, None], (1, 1, 3))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res*3])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        # No loss on anything >12A
        proximity_mask = gt_pair_dists < 12
        pair_dist_mask  = pair_dist_mask * proximity_mask

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask,
            dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) - num_res)

        se3_loss = trans_loss + rots_loss
        se3_vf_loss = trans_loss + rots_vf_loss
        full_dist_loss = bb_atom_loss + dist_mat_loss

        return noisy_batch, {
            "bb_atom_loss": bb_atom_loss,
            "trans_loss": trans_loss,
            "rots_loss": rots_loss,
            "se3_loss": se3_loss,
            "dist_mat_loss": dist_mat_loss,
            "full_dist_loss": full_dist_loss,
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
            try:
                mdtraj_metrics = metrics.calc_mdtraj_metrics(saved_path)
                ca_ca_metrics = metrics.calc_ca_ca_metrics(final_pos[:, CA_IDX])
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
        self._print_logger.info(f'Finished with eval epoch {self.current_epoch}')
        
    @torch.no_grad()
    def run_sampling(
            self,
            batch: Any,
            return_traj=False,
            return_model_outputs=False,
            num_timesteps=None,
            do_sde=None,
        ):
        if do_sde is None:
            do_sde = self._sampling_cfg.do_sde

        if not do_sde:
            return self.run_ode_sampling(
                batch, return_traj, return_model_outputs, num_timesteps)
        else:
            return self.run_sde_sampling(
                batch, return_traj, return_model_outputs, num_timesteps)

    @torch.no_grad()
    def run_ode_sampling(
        self,
        batch: Any,
        return_traj,
        return_model_outputs,
        num_timesteps,
        debug_trans: bool = False,
        debug_rots: bool = False,
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
        ).reshape(num_batch, num_res, 3, 3)
        
        prot_traj = [(trans_0, rots_0)]
        if num_timesteps is None:
            num_timesteps = self._sampling_cfg.num_timesteps
        ts = torch.linspace(self._sampling_cfg.min_t, 1.0, num_timesteps)
        if self._exp_cfg.rescale_time:
            ts = flow_utils.reschedule(ts)
        t_1 = ts[0]
        model_traj = []
        for t_2 in ts[1:]:
            d_t = t_2 - t_1
            trans_t_1, rots_t_1 = prot_traj[-1]
            with torch.no_grad():
                if self._exp_cfg.noise_trans and not debug_trans:
                    batch['trans_t'] = trans_t_1
                else:
                    batch['trans_t'] = batch['trans_1']
                if self._exp_cfg.noise_rots and not debug_rots:
                    batch['rotmats_t'] = rots_t_1
                else:
                    batch['rotmats_t'] = batch['rotmats_1']
                batch['t'] = torch.ones((num_batch, 1)).to(device) * t_1
                model_out = self.forward(batch)

            pred_trans_1 = model_out['pred_trans']
            pred_rots_1 = model_out['pred_rots']
            pred_rotmats_1 = pred_rots_1.get_rot_mats()
            pred_rots_vf = model_out['pred_rots_vf']

            model_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )
            trans_vf = (pred_trans_1 - trans_t_1) / (1 - t_1)
            trans_t_2 = trans_t_1 + trans_vf * d_t
            
            if self._model_cfg.predict_rot_vf:
                rots_t_2 = so3_utils.geodesic_t(
                    d_t / (1 - t_1), pred_rotmats_1, rots_t_1, rot_vf=pred_rots_vf)
            else:
                rots_t_2 = so3_utils.geodesic_t(
                    d_t / (1 - t_1), pred_rotmats_1, rots_t_1) 
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
        
    @torch.no_grad()
    def run_sde_sampling(
        self,
        batch: Any,
        return_traj,
        return_model_outputs,
        num_timesteps,
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
        ts = np.linspace(1.0 - (1/num_timesteps), self._sampling_cfg.min_t, num_timesteps)
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

                model_out = self.forward(batch)

            pred_trans_1 = model_out['pred_trans']
            pred_rots_1 = model_out['pred_rots'].get_rot_mats()
            pred_rots_vf = model_out['pred_rots_vf']

            model_traj.append(
                (pred_trans_1.detach().cpu(), pred_rots_1.detach().cpu())
            )
            assert d_t < 0

            # Euler-Maruyama step on the translations
            trans_score = ((1 - t_1) * pred_trans_1 - trans_t_1) / (t_1**2 * du.NM_TO_ANG_SCALE**2)
            trans_t_2 = (((-1)/(1-t_1)) * trans_t_1 - du.NM_TO_ANG_SCALE**2 * ((2 * t_1) / (1 - t_1)) * trans_score) * d_t + trans_t_1
            trans_t_2 = trans_t_2 + torch.randn_like(trans_t_2) * np.sqrt( (-d_t) * (2 * t_1) / (1 - t_1)  ) * du.NM_TO_ANG_SCALE

            # ODE step on the rotations

            # Not sure how to deal with this case
            rots_t_2 = so3_utils.geodesic_t(
                d_t / t_1, None, rots_t_1, rot_vf=pred_rots_vf) 

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
            model_out = self.forward(batch)

        pred_trans_1 = model_out['pred_trans']
        pred_rots_1 = model_out['pred_rots'].get_rot_mats()
        pred_rots_vf = model_out['pred_rots_vf']

        trans_t_2 = pred_trans_1
        rots_t_2 = pred_rots_1

        model_traj.append(
            (pred_trans_1.detach().cpu(), pred_rots_1.detach().cpu())
        )

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

    def _self_correct(self, noisy_batch):
        tmp_batch = copy.deepcopy(noisy_batch)
        with torch.no_grad():
            model_output = self.model(tmp_batch)
        tmp_batch['trans_1'] = model_output['pred_trans']
        tmp_batch['rotmats_1'] = model_output['pred_rotmats']
        correcting_batch = self._corrupt_batch(tmp_batch)
        noisy_batch['trans_t'] = correcting_batch['trans_t']
        noisy_batch['rotmats_t'] = correcting_batch['rotmats_t']
        noisy_batch['t'] = correcting_batch['t']
        return noisy_batch

    def training_step(self, batch: Any, stage: int):
        step_start_time = time.time()
        noisy_batch = self._corrupt_batch(batch)
        self_correct = False
        if self._exp_cfg.training.self_correcting and random.random() < 0.5:
            self_correct = True
            noisy_batch = self._self_correct(noisy_batch)
        self._log_scalar('train/self_correct', int(self_correct), prog_bar=False)
        _, batch_losses = self.model_step(noisy_batch)
        num_batch = batch_losses['bb_atom_loss'].shape[0]
        total_losses = {
            k: torch.mean(v) for k,v in batch_losses.items()
        }
        for k,v in total_losses.items():
            self._log_scalar(
                f"train/{k}", v, prog_bar=False, batch_size=num_batch)
        
        # Losses to track. Stratified across t.
        batch_t = torch.squeeze(noisy_batch['t'])
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
