import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from sim_suction_model.utils.pn2_utils.modules import PointNetSAModule, PointnetFPModule
from sim_suction_model.utils.pn2_utils.nn import SharedMLP
class Pointnet2_scorenet(nn.Module):
    """PointNet++ segmentation with single-scale grouping

    PointNetSA: PointNet Set Abstraction Layer
    PointNetFP: PointNet Feature Propagation Layer

    Args:
        score_classes (int): the number of grasp score classes
        num_centroids (tuple of int): the numbers of centroids to sample in each set abstraction module
        radius (tuple of float): a tuple of radius to query neighbours in each set abstraction module
        num_neighbours (tuple of int): the numbers of neighbours to query for each centroid
        sa_channels (tuple of tuple of int): the numbers of channels within each set abstraction module
        fp_channels (tuple of tuple of int): the numbers of channels for feature propagation (FP) module
        num_fp_neighbours (tuple of int): the numbers of nearest neighbor used in FP
        seg_channels (tuple of int): the numbers of channels in segmentation mlp
        dropout_prob (float): the probability to dropout input features

    """
    _SA_MODULE = PointNetSAModule
    _FP_MODULE = PointnetFPModule

    def __init__(self, input_chann = 6, k_score = 1, add_channel_flag=False, dropout_prob=0.5):
        super(Pointnet2_scorenet, self).__init__()
        self.k_score = k_score

        num_centroids     = (5120, 1024, 256)
        radius            = (0.02, 0.08, 0.2)
        num_neighbours    = (32, 32, 32)
        sa_channels       = ((128, 128, 256), (256, 256, 512), (512, 512, 1024))
        fp_channels       = ((1024, 1024), (512, 512), (256, 256, 256))
        num_fp_neighbours = (3, 3, 3)
        seg_channels      = (512, 256, 256, 128)

        num_sa_layers = len(num_centroids)
        num_fp_layers = len(fp_channels)

        # Set Abstraction Layers
        feature_channels = input_chann - 3 # 0
        self.sa_modules = nn.ModuleList()
        for ind in range(num_sa_layers):
            sa_module = self._SA_MODULE(in_channels=feature_channels,
                                        mlp_channels=sa_channels[ind],
                                        num_centroids=num_centroids[ind],
                                        radius=radius[ind],
                                        num_neighbours=num_neighbours[ind],
                                        use_xyz=True)
            self.sa_modules.append(sa_module)
            feature_channels = sa_channels[ind][-1]
        
        inter_channels = [input_chann - 3]# [0]
        inter_channels.extend([x[-1] for x in sa_channels])
        # Feature Propagation Layers
        self.fp_modules = nn.ModuleList()
        feature_channels = inter_channels[-1]
        for ind in range(num_fp_layers):
            fp_module = self._FP_MODULE(in_channels=feature_channels + inter_channels[-2 - ind],
                                        mlp_channels=fp_channels[ind],
                                        num_neighbors=num_fp_neighbours[ind])
            self.fp_modules.append(fp_module)
            feature_channels = fp_channels[ind][-1]

        # MLP
        if not add_channel_flag:
            self.mlp = SharedMLP(feature_channels, seg_channels, ndim=1, dropout_prob=dropout_prob)
        else:
            self.mlp = SharedMLP(feature_channels*3, seg_channels, ndim=1, dropout_prob=dropout_prob)
            

        self.conv_score = nn.Conv1d(seg_channels[-1], self.k_score, 1)
        self.bn_score = nn.BatchNorm1d(self.k_score)
        self.sigmoid = nn.Sigmoid()


    def forward(self, points, add_channel1=None, add_channel2=None):
        B,C,N = points.size()

        xyz = points[:,:3,:]
        feature = points[:,3:,:]

        # save intermediate results
        inter_xyz = [xyz]
        inter_feature = [feature]
        # Set Abstraction Layers
        for sa_module in self.sa_modules:
            xyz, feature = sa_module(xyz, feature)
            inter_xyz.append(xyz)
            inter_feature.append(feature)

        sparse_xyz = xyz
        sparse_feature = feature
        
        for fp_ind, fp_module in enumerate(self.fp_modules):
            dense_xyz = inter_xyz[-2 - fp_ind]
            dense_feature = inter_feature[-2 - fp_ind]
            fp_feature = fp_module(dense_xyz, sparse_xyz, dense_feature, sparse_feature)
            sparse_xyz = dense_xyz
            sparse_feature = fp_feature
            
        if add_channel1 is not None and add_channel2 is not None:
            add_channel1 = add_channel1.view(B,1,N).repeat(1,sparse_feature.shape[1],1)
            add_channel2 = add_channel2.view(B,1,N).repeat(1,sparse_feature.shape[1],1)
            sparse_feature = torch.cat((sparse_feature, add_channel1.float(), add_channel2.float()), dim=1)
 
        x = self.mlp(sparse_feature)
        x_score = self.bn_score(self.conv_score(x))
        x_score = x_score.transpose(2,1).contiguous()
        x_score = self.sigmoid(x_score)
        x_score = x_score.view(B, N, -1)

        return sparse_feature, x_score

if __name__ == '__main__':
    pass
