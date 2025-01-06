'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):  #只做平移，不做缩放
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    # m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    # pc = pc / m
    return pc


def farthest_point_sample(point,  npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]

    return point


class LiverVesselDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=False):
        # Initialize the class with the given parameters
        self.root = root  # The root directory where the data is stored
        self.npoints = args.num_point  # Number of points to be used (from arguments)
        self.process_data = process_data  # Flag to indicate if data should be processed
        self.uniform =  args.use_uniform_sample  # Whether to use uniform sampling or not
        self.use_normals = args.use_normals  # Whether to use normals in the data or not

        # Load the liver and vessel IDs for folders and their corresponding time IDs for files
        print(self.root)
        file_path = os.path.join(self.root, f'liverpoints_{split}.txt')
        
        # 添加路径检查
        if not os.path.exists(self.root):
            raise FileNotFoundError(f"数据目录不存在: {self.root}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        liver_entries = [line.rstrip().split() for line in open(file_path)]
        vessel_entries = [line.rstrip().split() for line in open(os.path.join(self.root, f'vesselspoints_{split}.txt'))]

        # Separate the entries into IDs and time IDs for liver and vessel
        self.liver_data = [(entry[0], entry[1]) for entry in liver_entries]  # (liver_id, time_id)
        self.vessel_data = [(entry[0], entry[1]) for entry in vessel_entries]  # (vessel_id, time_id)

        # Ensure that the number of liver and vessel data points are the same
        assert len(self.liver_data) == len(self.vessel_data), "Mismatch between liver and vessel data sizes!"

        print(f'The size of {split} data is {len(self.liver_data)}')


        # 检查是否需要处理数据
        if self.process_data:
            # 如果保存路径不存在，则需要处理数据
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)

                # 初始化数据点和标签的列表，长度等于数据路径的数量
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                # 遍历每个数据文件，通过索引访问数据路径列表中的每个数据
                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]  # 获取当前数据文件的信息
                    cls = self.classes[self.datapath[index][0]]  # 获取当前数据文件对应的类别
                    cls = np.array([cls]).astype(np.int32)  # 将类别转换为整数类型数组
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)  # 加载点集数据，并转换为浮点数类型

                    # 检查是否需要对点集进行均匀采样
                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)  # 使用最远点采样法采样
                    else:
                        point_set = point_set[0:self.npoints, :]  # 否则，只取前npoints个点

                    # 将处理后的点集和类别标签存储在列表中
                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                # 将点集和标签数据以二进制方式保存到指定路径的文件中
                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)

            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.liver_data)

    def _get_item(self, index):
        # Get the liver_id and time_id for liver data #目前两个id是相同的
        liver_id, liver_time_id  = self.liver_data[index]
        vessel_id, vessel_time_id = self.vessel_data[index]

        # Construct file paths for liver and vessel point clouds using their IDs and time IDs

        liver_d_file = os.path.join(self.root, f'liver_displacement_{liver_id}', f'{liver_time_id}.txt')

        vessel_d_file = os.path.join(self.root, f'vessels_displacement_{vessel_id}', f'{vessel_time_id}.txt')

        # Load the liver and vessel point sets from their respective text files
        liver = np.loadtxt(liver_d_file, delimiter=' ').astype(np.float32)
        vessel = np.loadtxt(vessel_d_file, delimiter=' ').astype(np.float32)


        if self.uniform:
            liver_points = farthest_point_sample(liver, self.npoints)
            vessel_points = farthest_point_sample(vessel,  self.npoints)
        liver_points = liver
        vessel_points = vessel

        #点云数据中心化
        liver_points = pc_normalize(liver_points)
        vessel_points = pc_normalize(vessel_points)

        if not self.use_normals:
            liver_points = liver_points[:, 0:6]
            vessel_points = vessel_points[:, 0:6]
            vessel_point = vessel_points[:, 0:3]

        else:
            vessel_point = np.concatenate((vessel_points[:, :3], vessel_points[:, 6:]), axis=1)  # 组合点云和法向量，去掉位移
        vessel_displacement = vessel_points[: ,3:6] #label
      #  print('the original displacement shape is:', vessel_displacement.shape)
        vessel_displacement = farthest_point_sample(vessel_displacement, 256)
      #  print('the displacement shape is:',vessel_displacement.shape)


        return liver_points, vessel_point, vessel_displacement

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = LiverVesselDataLoader('F:\Project\pointMLP-pytorch-main\pointmlp_displacement\data\data_1',split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for liver_points, vessel_point, vessel_displacement in DataLoader:
        print(f'Liver points shape: {liver_points.shape}')
        print(f'Vessel points shape: {vessel_point.shape}')
        print(f'Vessel deformation shape: {vessel_displacement.shape}')
