import torch
import torch.nn as nn
import torch.optim as optim
import sys
import torch.nn.functional as F
from torch.cuda import amp
from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import os
import argparse
import datetime

from LIAF import *
from LIAFResNet import *
from LIAFResNet_18 import Config

def eval(model, test_loader, device):
    model.eval()
    acn = num = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.transpose(2, 1)
            x, y = x.to(device), y.to(device)
            output = model(x)
            # 计算分类正确的数量, 并累加样本数量
            pred = torch.argmax(output, dim=1)
            acn += torch.eq(pred, y).sum().item()
            num += y.shape[0]
    return acn / num

def main():
    # 修改数据集地址、加载权重的地址
    dataset_pth = '/root/autodl-tmp/DVS128/'
    weight_pth = './resnet_greedysoup.pkl'
    batch_size = 8
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据集
    test_dataset = DVS128Gesture(root='/root/autodl-tmp/DVS128/', train=False, data_type='frame', frames_number=16, split_by='number')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
    
    # 创建模型
    config = Config()
    model = LIAFResNet(config)

    # 加载预训练权重
    checkpoint = torch.load(weight_pth, map_location=device)
    model_dict = model.state_dict()
    # 只保留和模型内的参数匹配的参数权重
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model = model.to(device)
    
    print('Testing')
    acc = eval(model, test_loader, device)
    print(f'此权重在测试集上的准确率为: {(acc * 100):.3f}')


if __name__ == '__main__':
    main()