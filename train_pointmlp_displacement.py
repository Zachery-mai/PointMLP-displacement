"""
基于PointMLP的点云位移预测训练脚本
"""
import os
import sys
import torch
import numpy as np
import datetime
import logging
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.liver_dataset import LiverVesselDataLoader

def parse_args():
    parser = argparse.ArgumentParser('PointMLP Displacement Training')
    parser.add_argument('--use_cpu', action='store_true', default=True, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=6, help='batch size in training')
    parser.add_argument('--model', default='pointmlp_displacement', help='model name')
    parser.add_argument('--epoch', default=50, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
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
            mean_loss.append(loss.item())
            
    return np.mean(mean_loss)

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    # 创建实验目录
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('displacement')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    # 创建日志记录器
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 加载数据
    data_path = 'data/data_1/'
    train_dataset = LiverVesselDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = LiverVesselDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 加载模型
    model = importlib.import_module(args.model)
    classifier = model.get_model(args.num_point)
    criterion = model.get_loss()

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    # 优化器设置
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    
    # 训练循环
    best_test_loss = float("inf")
    global_epoch = 0
    
    for epoch in range(args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        
        # 训练阶段
        classifier.train()
        for batch_id, (points_l, points_v, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader)):
            optimizer.zero_grad()
            
            if not args.use_cpu:
                points_l, points_v, target = points_l.cuda(), points_v.cuda(), target.cuda()
                
            pred = classifier(points_l, points_v)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            
        scheduler.step()
        
        # 测试阶段
        test_loss = test(classifier.eval(), testDataLoader, criterion)
        log_string('Test Loss: %f' % test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            log_string('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'test_loss': test_loss,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
        global_epoch += 1

if __name__ == '__main__':
    args = parse_args()
    main(args) 