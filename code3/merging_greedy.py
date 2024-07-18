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
from models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
test_dataset = datasets.CIFAR10(root='./dataset', train=False, transform=transform_test, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=265, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
    
def load_model_weights(file_path):
    return torch.load(file_path, map_location=device)

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

def get_model_from_sd(state_dict):
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
    # 根据状态字典加载模型
    model.load_state_dict(state_dict)
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
    accuracy = float(parts[-2])  # 倒数第二部分是准确率
    models_with_acc.append((file_path, accuracy))

# 按准确率从高到低排序模型
models_with_acc.sort(key=lambda x: x[1], reverse=True)
models_with_acc = models_with_acc[:]
# print(models_with_acc[:])s

# 用准确率最高的模型初始化贪心汤
best_model_path, best_val_acc_so_far = models_with_acc[0]
greedy_soup_ingredients = [best_model_path]
greedy_soup_params = load_model_weights(best_model_path)

# 遍历剩余的模型
for i in range(1, len(models_with_acc)):
    print(f'测试模型 {i} / {len(models_with_acc)-1}')
    
    # 加载新模型的参数
    new_model_path, new_model_acc = models_with_acc[i]
    new_ingredient_params = load_model_weights(new_model_path)
    
    # 计算潜在的贪心汤参数
    num_ingredients = len(greedy_soup_ingredients)
    potential_greedy_soup_params = {
        k: greedy_soup_params[k].clone() * (num_ingredients / (num_ingredients + 1.)) + 
           new_ingredient_params[k].clone() * (1. / (num_ingredients + 1))
        for k in new_ingredient_params
    }
    
    # 使用潜在的贪心汤参数创建模型并在验证集上测试
    model = get_model_from_sd(potential_greedy_soup_params)
    held_out_val_accuracy = eval(model, test_loader, device)
    held_out_val_accuracy = held_out_val_accuracy * 100
    
    # 如果潜在贪心汤模型的准确率更高，则更新贪心汤
    print(f'{new_model_acc}, 潜在贪心汤验证集准确率: {held_out_val_accuracy:.3f}, 当前最佳: {best_val_acc_so_far:.3f}')
    if held_out_val_accuracy >= best_val_acc_so_far:
        greedy_soup_ingredients.append(new_model_path)
        best_val_acc_so_far = held_out_val_accuracy
        greedy_soup_params = potential_greedy_soup_params
        print(f'添加到贪心汤中')

# 保存
model = get_model_from_sd(greedy_soup_params)
torch.save(model.state_dict(), './Spike-DrivenV2_CIFAR10_greedysoup.pkl')
print('合并模型已保存')

