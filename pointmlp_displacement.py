import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointmlp import LocalGrouper, ConvBNReLU1D, PreExtraction, PosExtraction, PointNetFeaturePropagation

class PointMLPDisplacement(nn.Module):
    def __init__(self, points=1024, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[4, 4, 4, 4],
                 de_dims=[512, 256, 128, 64], de_blocks=[2,2,2,2],
                 gmp_dim=64, **kwargs):
        super(PointMLPDisplacement, self).__init__()
        self.stages = len(pre_blocks)
        self.points = points
        
        # 初始特征嵌入层，输入维度为3(xyz坐标)
        self.embedding_liver = ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation)
        self.embedding_vessel = ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation)
        
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
            
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        
        last_channel = embed_dim
        anchor_points = self.points
        en_dims = [last_channel]
        
        ### Building Encoder #####
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            
            # append local_grouper_list
            local_grouper = LocalGrouper(last_channel, anchor_points, kneighbor, use_xyz, normalize)
            self.local_grouper_list.append(local_grouper)
            
            # append pre_block_list
            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=groups,
                                           res_expansion=res_expansion,
                                           bias=bias, activation=activation, use_xyz=use_xyz)
            self.pre_blocks_list.append(pre_block_module)
            
            # append pos_block_list
            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups,
                                           res_expansion=res_expansion, bias=bias, activation=activation)
            self.pos_blocks_list.append(pos_block_module)
            
            last_channel = out_channel
            en_dims.append(last_channel)

        ### Building Decoder #####
        self.decode_list = nn.ModuleList()
        en_dims.reverse()
        de_dims.insert(0,en_dims[0])
        assert len(en_dims) == len(de_dims) == len(de_blocks)+1
        
        for i in range(len(en_dims)-1):
            self.decode_list.append(
                PointNetFeaturePropagation(de_dims[i]+en_dims[i+1], de_dims[i+1],
                                         blocks=de_blocks[i], groups=groups, res_expansion=res_expansion,
                                         bias=bias, activation=activation)
            )

        # 最终输出层 - 预测3D位移
        self.displacement_head = nn.Sequential(
            nn.Conv1d(de_dims[-1], 128, 1, bias=bias),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 3, 1, bias=bias)  # 输出3D位移向量
        )

    def forward(self, liver_points, vessel_points):
        # 输入点云的形状应该是 [B, 3, N]
        xyz_liver = liver_points.permute(0, 2, 1)  # [B, N, 3]
        xyz_vessel = vessel_points.permute(0, 2, 1)  # [B, N, 3]
        
        # 特征嵌入
        x_liver = self.embedding_liver(liver_points)  # [B, D, N]
        x_vessel = self.embedding_vessel(vessel_points)  # [B, D, N]
        
        # 合并特征
        x = x_liver + x_vessel
        xyz = xyz_liver  # 使用肝脏点云作为参考坐标系
        
        xyz_list = [xyz]
        x_list = [x]
        
        # Encoder
        for i in range(self.stages):
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))
            x = self.pre_blocks_list[i](x)
            x = self.pos_blocks_list[i](x)
            xyz_list.append(xyz)
            x_list.append(x)
            
        # Decoder
        xyz_list.reverse()
        x_list.reverse()
        x = x_list[0]
        
        for i in range(len(self.decode_list)):
            x = self.decode_list[i](xyz_list[i+1], xyz_list[i], x_list[i+1], x)
            
        # 预测位移
        displacement = self.displacement_head(x)
        displacement = displacement.permute(0, 2, 1)  # [B, N, 3]
        
        return displacement

def get_model(num_points=1024, normal_channel=False):
    return PointMLPDisplacement(points=num_points)

def get_loss():
    return nn.MSELoss()