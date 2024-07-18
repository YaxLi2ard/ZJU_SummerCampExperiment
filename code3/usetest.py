import torch
import torch.nn as nn
import torch.optim as optim
import sys
from torchvision import transforms, datasets
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
            x, y = x.to(device), y.to(device)
            output = model(x)
            functional.reset_net(model)
            # 计算分类正确的数量, 并累加样本数量
            pred = torch.argmax(output, dim=1)
            acn += torch.eq(pred, y).sum().item()
            num += x.shape[0]
    return acn / num

def main():
    # 修改数据集地址和加载权重的地址
    dataset_pth = './dataset'
    weight_pth = './weights/Spike-DrivenV2_90.5_.pkl'
    batch_size = 265
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据集
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    test_dataset = datasets.CIFAR10(root=dataset_pth, train=False, transform=transform_test, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
    
    # 创建模型
    model = Spiking_vit_MetaFormer(
        img_size_h=32,
        img_size_w=32,
        patch_size=8,
        embed_dim=[128, 256, 512, 640],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=11,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=8,
        sr_ratios=1,
    )
    model.T = 5
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