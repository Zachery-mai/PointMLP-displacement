"""
基于PointMLP的点云位移预测测试脚本
"""
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
from data_utils.liver_dataset import LiverVesselDataLoader

def parse_args():
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()

def test(model, loader, criterion):
    mean_loss = []
    model.eval()
    
    for j, (points_l, points_v, target) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points_l, points_v, target = points_l.cuda(), points_v.cuda(), target.cuda()
            
        with torch.no_grad():
            pred = model(points_l, points_v)
            loss = criterion(pred, target)
            
            # 计算每个点的位移误差
            displacement_error = torch.norm(pred - target, dim=2)  # [B, N]
            mean_error = displacement_error.mean().item()
            mean_loss.append(mean_error)
            
    return np.mean(mean_loss)

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 创建日志目录
    experiment_dir = 'log/displacement/' + args.log_dir

    # 设置日志
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    # 加载数据
    data_path = 'data/data_1/'
    test_dataset = LiverVesselDataLoader(root=data_path, args=args, split='test', process_data=False)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    # 加载模型
    model = importlib.import_module('pointmlp_displacement')
    classifier = model.get_model(num_points=args.num_point)
    criterion = model.get_loss()
    
    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    # 加载预训练模型
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    # 测试
    with torch.no_grad():
        mean_loss = test(classifier.eval(), testDataLoader, criterion)
        log_string('Mean Displacement Error: %f' % mean_loss)

if __name__ == '__main__':
    args = parse_args()
    main(args) 