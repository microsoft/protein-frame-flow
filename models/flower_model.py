import torch
from torch import nn

from models.node_feature_net import NodeFeatureNet
from models.edge_feature_net import EdgeFeatureNet
from models.pair_network import PairTransformNet
from models.structure_network import StructureNet

from data import utils as du
from data import so3_utils


class Flower(nn.Module):

    def __init__(self, model_cfg):
        super(Flower, self).__init__()

        self.node_feature_net = NodeFeatureNet(model_cfg.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_cfg.edge_features)
        
        self.pair_transform_net = PairTransformNet(model_cfg.pair_network)
        self.structure_net = StructureNet(model_cfg.structure_network)

    def forward(self, input_feats):
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        init_node_embed = self.node_feature_net(
            input_feats['t'], node_mask)
        init_edge_embed = self.edge_feature_net(
            init_node_embed, input_feats['trans_t'], edge_mask)
        edge_embed = self.pair_transform_net(
            init_edge_embed, edge_mask)

        trans_t = input_feats['trans_t']
        rotmats_t = input_feats['rotmats_t']
        init_frames = du.create_rigid(rotmats_t, trans_t)
        final_frames = self.structure_net(
            init_node_embed, edge_embed, init_frames, node_mask)

        pred_trans = final_frames.get_trans()
        pred_rots = final_frames.get_rots().get_rot_mats().type(torch.float32)
        rots_vf = so3_utils.rot_vf(rotmats_t, pred_rots)

        return {
            'pred_trans': pred_trans,
            'pred_rotmats': pred_rots,
            'pred_rots_vf': rots_vf,
        }