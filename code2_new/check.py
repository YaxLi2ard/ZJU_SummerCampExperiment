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
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
    
def load_model_weights(file_path):
    return torch.load(file_path, map_location=device)

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

def condition(k):
    # if 'block3' in k or 'block4' in k or 'downsample4' in k:
    #     return 0
    return 1

# 模型权重文件所在的目录
weights_dir = './weights'
# 获取目录中所有pkl格式的权重文件
weights_files = glob.glob(os.path.join(weights_dir, '*.pkl'))

# 从文件名中提取模型准确率并进行排序
models_with_acc = []
for file_path in weights_files:
    filename = os.path.basename(file_path)
    parts = filename.split('_')
    accuracy = float(parts[-2])  # 倒数第二部分是准确率
    models_with_acc.append((file_path, accuracy))

# 按准确率从高到低排序模型
models_with_acc.sort(key=lambda x: x[1], reverse=True)
models_with_acc = models_with_acc[:]
# print(models_with_acc[:])

# 用准确率最高的模型初始化贪心汤
best_model_path, best_val_acc_so_far = models_with_acc[0]
greedy_soup_ingredients = [best_model_path]
greedy_soup_params = load_model_weights(best_model_path)

flag = [0]

# 遍历剩余的模型
for i in range(1, len(models_with_acc)):
    print(f'\n测试模型 {i} / {len(models_with_acc)-1}')
    
    # 加载新模型的参数
    new_model_path, new_model_acc = models_with_acc[i]
    new_ingredient_params = load_model_weights(new_model_path)
    
    # 计算潜在的贪心汤参数
    num_ingredients = len(greedy_soup_ingredients)
    potential_greedy_soup_params = {
        k: greedy_soup_params[k].clone() * (num_ingredients / (num_ingredients + 1.)) + 
           new_ingredient_params[k].clone() * (1. / (num_ingredients + 1))
        if condition(k) else greedy_soup_params[k]
        for k in new_ingredient_params
    }
    
    # 使用潜在的贪心汤参数创建模型并在验证集上测试
    model = get_model_from_sd(potential_greedy_soup_params)
    held_out_val_accuracy = eval(model, test_loader, device)
    held_out_val_accuracy = held_out_val_accuracy * 100
    
    # 如果合并后比原来的准确率还低，则标为无效，否则有效
    print(f'{new_model_acc}, 潜在贪心汤验证集准确率: {held_out_val_accuracy:.3f}, 当前最佳: {best_val_acc_so_far:.3f}')
    if held_out_val_accuracy <= new_model_acc - 2:
        print(f'无效')
    else:
        flag.append(i)
        print('有效')
    
flag = np.array(flag)
np.save('flag.npy', flag)