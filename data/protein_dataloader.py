"""Protein data loader."""
import math
import torch
import logging
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler, dist


class ProteinData(LightningDataModule):

    def __init__(self, *, data_cfg, train_dataset, valid_dataset, predict_dataset=None):
        super().__init__()
        self.data_cfg = data_cfg
        self.loader_cfg = data_cfg.loader
        self.sampler_cfg = data_cfg.sampler
        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset
        self._predict_dataset = predict_dataset

    def train_dataloader(self, rank=None, num_replicas=None):
        num_workers = self.loader_cfg.num_workers
        return DataLoader(
            self._train_dataset,
            batch_sampler=LengthBatcher(
                sampler_cfg=self.sampler_cfg,
                metadata_csv=self._train_dataset.csv,
                rank=rank,
                num_replicas=num_replicas,
            ),
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self._valid_dataset,
            sampler=DistributedSampler(self._valid_dataset, shuffle=False),
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        num_workers = self.loader_cfg.num_workers
        return DataLoader(
            self._predict_dataset,
            sampler=DistributedSampler(self._predict_dataset, shuffle=False),
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
            persistent_workers=True,
        )


class LengthBatcher:

    def __init__(
            self,
            *,
            sampler_cfg,
            metadata_csv,
            seed=123,
            shuffle=True,
            num_replicas=None,
            rank=None,
        ):
        super().__init__()
        self._log = logging.getLogger(__name__)
        if num_replicas is None:
            self.num_replicas = dist.get_world_size()
        else:
            self.num_replicas = num_replicas
        if rank is None:
            self.rank = dist.get_rank()
        else:
            self.rank = rank

        self._sampler_cfg = sampler_cfg
        self._data_csv = metadata_csv

        # Each replica needs the same number of batches. We set the number
        # of batches to arbitrarily be the number of examples per replica.
        if 'cluster' in self._data_csv:
            num_batches = self._data_csv['cluster'].nunique()
        else:
            num_batches = len(self._data_csv)
        self._num_batches = math.ceil(num_batches / self.num_replicas)
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.max_batch_size =  self._sampler_cfg.max_batch_size
        self._log.info(f'Created dataloader rank {self.rank+1} out of {self.num_replicas}')

    def _sample_indices(self):
        if 'cluster' in self._data_csv:
            cluster_sample = self._data_csv.groupby('cluster').sample(
                1, random_state=self.seed + self.epoch)
            return cluster_sample['index'].tolist()
        else:
            return self._data_csv['index'].tolist()
        
    def _replica_epoch_batches(self):
        # Make sure all replicas share the same seed on each epoch.
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)
        indices = self._sample_indices()
        if self.shuffle:
            new_order = torch.randperm(len(indices), generator=rng).numpy().tolist()
            indices = [indices[i] for i in new_order]

        if len(self._data_csv) > self.num_replicas:
            replica_csv = self._data_csv.iloc[indices[self.rank::self.num_replicas]]
        else:
            replica_csv = self._data_csv
        
        # Each batch contains multiple proteins of the same length.
        sample_order = []
        for seq_len, len_df in replica_csv.groupby('modeled_seq_len'):
            max_batch_size = min(
                self.max_batch_size,
                self._sampler_cfg.max_num_res_squared // seq_len**2 + 1,
            )
            num_batches = math.ceil(len(len_df) / max_batch_size)
            for i in range(num_batches):
                batch_df = len_df.iloc[i*max_batch_size:(i+1)*max_batch_size]
                batch_indices = batch_df['index'].tolist()
                batch_repeats = math.floor(max_batch_size / len(batch_indices))
                sample_order.append(batch_indices * batch_repeats)
        
        # Remove any length bias.
        if self.shuffle:
            new_order = torch.randperm(len(sample_order), generator=rng).numpy().tolist()
            return [sample_order[i] for i in new_order]
        return sample_order

    def _create_batches(self):
        # Make sure all replicas have the same number of batches Otherwise leads to bugs.
        # See bugs with shuffling https://github.com/Lightning-AI/lightning/issues/10947
        all_batches = []
        num_augments = -1
        while len(all_batches) < self._num_batches:
            all_batches.extend(self._replica_epoch_batches())
            num_augments += 1
            if num_augments > 1000:
                raise ValueError('Exceeded number of augmentations.')
        if len(all_batches) >= self._num_batches:
            all_batches = all_batches[:self._num_batches]
        self.sample_order = all_batches

    def __iter__(self):
        self._create_batches()
        self.epoch += 1
        return iter(self.sample_order)

    def __len__(self):
        return len(self.sample_order)
