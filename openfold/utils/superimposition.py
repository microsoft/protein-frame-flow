# Copyright 2021 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from Bio.SVDSuperimposer import SVDSuperimposer
import numpy as np
import torch


def _superimpose_np(reference, coords):
    """
        Superimposes coordinates onto a reference by minimizing RMSD using SVD.

        Args:
            reference:
                [N, 3] reference array
            coords:
                [N, 3] array
        Returns:
            A tuple of [N, 3] superimposed coords and the final RMSD.
    """
    sup = SVDSuperimposer()
    sup.set(reference, coords)
    sup.run()
    return sup


def _superimpose_single(reference, coords):
    reference_np = reference.detach().cpu().numpy()    
    coords_np = coords.detach().cpu().numpy()
    sup = _superimpose_np(reference_np, coords_np)
    superimposed, rmsd = sup.get_transformed(), sup.get_rms()
    rot, tran = sup.get_rotran()
    return coords.new_tensor(superimposed), coords.new_tensor(rmsd), coords.new_tensor(rot), coords.new_tensor(tran)


def superimpose(reference, coords, return_transform=False):
    """
        Superimposes coordinates onto a reference by minimizing RMSD using SVD.

        Args:
            reference:
                [*, N, 3] reference tensor
            coords:
                [*, N, 3] tensor
        Returns:
            A tuple of [*, N, 3] superimposed coords and [*] final RMSDs.
    """
    batch_dims = reference.shape[:-2]
    flat_reference = reference.reshape((-1,) + reference.shape[-2:])
    flat_coords = coords.reshape((-1,) + reference.shape[-2:])
    superimposed_list = []
    rmsds = []
    rots_list = []
    trans_list = []
    for r, c in zip(flat_reference, flat_coords):
       superimposed, rmsd, rot, tran = _superimpose_single(r, c)
       rots_list.append(rot)
       trans_list.append(tran)
       superimposed_list.append(superimposed)
       rmsds.append(rmsd)

    superimposed_stacked = torch.stack(superimposed_list, dim=0)
    rmsds_stacked = torch.stack(rmsds, dim=0)
    rots_stacked = torch.stack(rots_list, dim=0)
    trans_stacked = torch.stack(trans_list, dim=0)

    superimposed_reshaped = superimposed_stacked.reshape(
        batch_dims + coords.shape[-2:]
    )
    rmsds_reshaped = rmsds_stacked.reshape(
        batch_dims
    )
    rots_reshaped = rots_stacked.reshape(
        batch_dims + (3, 3))
    tran_reshaped = trans_stacked.reshape(
        batch_dims + (3,))
    if return_transform:
        return superimposed_reshaped, rmsds_reshaped, rots_reshaped, tran_reshaped
    return superimposed_reshaped, rmsds_reshaped
