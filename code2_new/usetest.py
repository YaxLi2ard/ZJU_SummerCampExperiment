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

from models import *

def eval(model, test_loader, device):
    model.eval()
    acn = num = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.transpose(1, 0)
            x, y = x.to(device), y.to(device)
            output = model(x)
            functional.reset_net(model)
            # 计算分类正确的数量, 并累加样本数量
            pred = torch.argmax(output, dim=1)
            acn += torch.eq(pred, y).sum().item()
            num += y.shape[0]
    return acn / num

def main():
    # 修改数据集地址和加载权重的地址
    dataset_pth = '/root/autodl-tmp/DVS128/'
    weight_pth = './Spike-DrivenV2_DVS128_greedysoup.pkl'
    batch_size = 6
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据集
    test_dataset = DVS128Gesture(root='/root/autodl-tmp/DVS128/', train=False, data_type='frame', frames_number=16, split_by='number')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
    
    # 创建模型
    model = Spiking_vit_MetaFormer_lite(
        embed_dim=[128, 256, 512, 640],  # 96, 192, 384, 480
        num_heads=8,
        addition=True,
        mlp_drop_rate=0.1,
        head_drop_rate=0.1,
        mlp_ratios=4,
        in_channels=2,
        num_classes=11,
    )
    # model.T = 8
    checkpoint = torch.load(weight_pth, map_location=device)
    model_dict = model.state_dict()
    # 只保留和模型内的参数匹配的参数权重
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model = model.to(device)
    
    print('Testing')
    acc = eval(model, test_loader, device)
    print(f'此权重在测试集上的准确率为: {(acc * 100):.5f}')


if __name__ == '__main__':
    main()