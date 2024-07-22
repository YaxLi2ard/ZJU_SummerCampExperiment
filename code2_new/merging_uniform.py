import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import sys
import glob
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
import numpy as np
from models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_dataset = DVS128Gesture(root='/root/autodl-tmp/DVS128/', train=False, data_type='frame', frames_number=16, split_by='number')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
    
def load_model_weights(file_path):
    checkpoint = torch.load(file_path, map_location=device)
    model = get_model_from_sd(checkpoint)
    return model.state_dict()

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

def get_model_from_sd(checkpoint):
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
    # model.T = 5
    # 根据状态字典加载模型
    model_dict = model.state_dict()
    # 只保留和模型内的参数匹配的参数权重
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = model.to(device)
    return model

# 模型权重文件所在的目录
weights_dir = './weights'
# 获取目录中所有pkl格式的权重文件
weights_files = glob.glob(os.path.join(weights_dir, '*.pkl'))

# 从文件名中提取模型准确率并进行排序
models_with_acc = []
for file_path in weights_files:
    filename = os.path.basename(file_path)
    parts = filename.split('_')
    accuracy = float(parts[-2])  # 假设倒数第二部分是准确率
    models_with_acc.append((file_path, accuracy))

# 按准确率从高到低排序模型
models_with_acc.sort(key=lambda x: x[1], reverse=True)
fl = np.load('flag.npy')
fl = fl.tolist()
models_with_acc = [models_with_acc[i] for i in fl]

# 合并所有模型的权重，计算均值
uniform_soup = None
num_models = len(models_with_acc)

for j, (model_path, accuracy) in enumerate(models_with_acc):
    print(f'将模型 {j + 1} / {num_models} 添加到均匀汤中.')
    
    # 加载模型权重
    state_dict = load_model_weights(model_path)
    
    if j == 0:
        # 初始化均匀汤
        uniform_soup = {k: v * (1. / num_models) for k, v in state_dict.items()}
    else:
        # 叠加权重并计算平均值
        uniform_soup = {k: v * (1. / num_models) + uniform_soup[k] for k, v in state_dict.items()}

# 使用均匀汤权重创建模型
model = get_model_from_sd(uniform_soup)
acc = eval(model, test_loader, device)
print(f'平均合并最终准确率: {(acc * 100):.3f}')

torch.save(model.state_dict(), './Spike-DrivenV2_DVS128_uniformsoup.pkl')
print('合并模型已保存')