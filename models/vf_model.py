
import torch
import math
from torch import nn

from models import ipa_pytorch
from data import utils as du
from data import so3_utils


def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size//2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size)))
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size)))
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding

def get_timestep_embedding(timesteps, embedding_dim, max_positions=1000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = nn.F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class VFModel(nn.Module):

    def __init__(self, model_conf):
        super(VFModel, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * 0.1)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * 10.0)
        
        # Initial node embeddings
        node_feat_size = self._model_conf.t_embed_size + self._model_conf.index_embed_size
        node_embed_size = self._model_conf.node_embed_size
        self.node_embedder = nn.Sequential(
            nn.Linear(node_feat_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),
        )

        # Initial edge embeddings
        edge_embed_size = self._model_conf.edge_embed_size
        self.edge_embedder = nn.Sequential(
            nn.Linear(node_embed_size * 2 + self._model_conf.index_embed_size, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.LayerNorm(edge_embed_size),
        )
        if self._model_conf.predict_rot_vf:
            self._rot_vf_head = ipa_pytorch.BackboneUpdate(
                self._model_conf.node_embed_size, False)

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            if self._ipa_conf.use_skip:
                tfmr_in += self._ipa_conf.c_skip
                self.trunk[f'skip_embed_{b}'] = ipa_pytorch.Linear(
                    self._model_conf.node_embed_size,
                    self._ipa_conf.c_skip,
                    init="final"
                )
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, self._model_conf.use_rot_updates)

            if b < self._ipa_conf.num_blocks-1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res**2, -1])

    def forward(self, input_feats):

        # Initial node and edge features
        node_mask = input_feats['res_mask'].type(torch.float32)
        device = node_mask.device
        num_batch = node_mask.shape[0]
        num_res = node_mask.shape[1]
        node_indices = torch.arange(num_res, device=device)
        index_embed = get_index_embedding(
            node_indices,
            self._model_conf.index_embed_size
        )[None].repeat(num_batch, 1, 1)
        t_embed = get_timestep_embedding(
            input_feats['t'][:, 0],
            self._model_conf.t_embed_size
        )[:, None, :].repeat(1, num_res, 1)
        init_node_feats = torch.concat(
            [index_embed, t_embed], dim=-1)
        init_node_embed = self.node_embedder(init_node_feats)

        # Initial edge embeddings
        rel_seq_offset = node_indices[:, None] - node_indices[None, :]
        rel_seq_offset = rel_seq_offset.flatten()
        rel_seq_embed = get_index_embedding(
            rel_seq_offset,
            self._model_conf.index_embed_size
        )[None].repeat(num_batch, 1, 1)
        cross_node_embed = self._cross_concat(init_node_embed, num_batch, num_res)
        init_edge_feats = torch.concat([
           cross_node_embed, rel_seq_embed
        ], dim=-1)
        init_edge_embed = self.edge_embedder(init_edge_feats)
        init_edge_embed = init_edge_embed.reshape([num_batch, num_res, num_res, -1])
        edge_mask = node_mask[..., None] * node_mask[..., None, :]

        # Initial rigids
        trans_t = input_feats['trans_t']
        rotmats_t = input_feats['rotmats_t']
        curr_rigids = du.create_rigid(rotmats_t, trans_t)

        # Main trunk
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            if self._ipa_conf.use_skip:
                seq_tfmr_in = torch.cat([
                    node_embed, self.trunk[f'skip_embed_{b}'](init_node_embed)
                ], dim=-1)
            else:
                seq_tfmr_in = node_embed
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                seq_tfmr_in, src_key_padding_mask=1 - node_mask)
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            if self._model_conf.use_rot_updates:
                curr_rigids = curr_rigids.compose_q_update_vec(
                    rigid_update, node_mask[..., None])
            else:
                curr_rigids = curr_rigids.compose_tran_update_vec(
                    rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rots = curr_rigids.get_rots().get_rot_mats()
        if self._model_conf.predict_rot_vf:
            rots_vf = self._rot_vf_head(node_embed)
        else:
            rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rots)
        rots_vf *= node_mask[..., None]

        return {
            'pred_trans': pred_trans,
            'pred_rotmats': pred_rots,
            'pred_rots_vf': rots_vf,
        }
