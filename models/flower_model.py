import torch
from torch import nn

from models.node_feature_net import NodeFeatureNet
from models.edge_feature_net import EdgeFeatureNet
from models.pair_network import PairTransformNet
from models.potential_net import PotentialNet
from models import ipa_pytorch

from data import utils as du
from data import so3_utils


class Flower(nn.Module):

    def __init__(self, model_cfg):
        super(Flower, self).__init__()
        self._model_cfg = model_cfg

        self.node_feature_net = NodeFeatureNet(model_cfg.node_features)
        # TODO: Make this symmetric.
        self.edge_feature_net = EdgeFeatureNet(model_cfg.edge_features)
        self.potential_net = PotentialNet(model_cfg.potential_network)
        if model_cfg.predict_rot_vf:
            self._rot_vf_head = ipa_pytorch.BackboneUpdate(
                model_cfg.node_embed_size, False)

    def forward(self, input_feats):
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        continuous_t = input_feats['t']
        discrete_t = torch.floor(1000 * continuous_t)

        # Initialize node and edge embeddings
        init_node_embed = self.node_feature_net(
            discrete_t, node_mask)
        init_edge_embed = self.edge_feature_net(
            init_node_embed, input_feats['trans_t'], edge_mask)

        # Initialize frames
        trans_t = input_feats['trans_t']
        rotmats_t = input_feats['rotmats_t']
        init_frames = du.create_rigid(rotmats_t, trans_t)

        # Embed and update structure
        node_embed, final_frames = self.potential_net(
            init_node_embed, init_edge_embed, init_frames, 
            node_mask, edge_mask)

        pred_trans = final_frames.get_trans()
        pred_rots = final_frames.get_rots().get_rot_mats().type(torch.float32)
        if self._model_cfg.predict_rot_vf:
            rots_vf = self._rot_vf_head(node_embed)
        else:
            rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rots)

        return {
            'pred_trans': pred_trans,
            'pred_rotmats': pred_rots,
            'pred_rots_vf': rots_vf,
        }