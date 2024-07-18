import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
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

def main(config_dic):
    print('\n\n参数 -> ', config_dic)
    # 部分超参数设置
    learning_rate = config_dic['lr']
    batch_size = config_dic['batch_size']
    epochs = config_dic['epochs']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 数据增强
    transform_train = transforms.Compose([
        # transforms.ColorJitter(brightness=(2,2), contrast=(0.5,0.5), saturation=(0.5,0.5)),
        transforms.RandomCrop(32, padding=6),  # 对原始32*32图像四周各填充4个0像素（40*40），然后随机裁剪成32*32
        transforms.RandomHorizontalFlip(p=0.6),  # 按0.5的概率水平翻转图片
        # transforms.RandomRotation(degrees=(8, 8)),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.5, 2)),  # 随机遮挡
    ])
 
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # 加载数据集
    train_dataset = datasets.CIFAR10(root='./dataset', train=True, transform=transform_train, download=True)
    test_dataset = datasets.CIFAR10(root='./dataset', train=False, transform=transform_test, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=False)
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
    
    checkpoint = torch.load('55M_kd.pth', map_location=device)
    checkpoint = checkpoint['model']
    
    model_dict = model.state_dict()
    # 只保留和模型内的参数匹配的参数权重
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    if config_dic['optim'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=config_dic['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.6, weight_decay=config_dic['weight_decay'])
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=10)

    # 训练
    print('---------------------------------Training---------------------------------')
    start_time = time.time()
    acn = num = 0
    for epoch in range(epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            model.zero_grad()
            optimizer.zero_grad()
            output = model(x)

            loss = criterion(output, y)

            loss.backward()
            optimizer.step()
            # torch.cuda.synchronize()
            functional.reset_net(model)

            # 计算分类正确的数量, 并累加样本数量
            pred = torch.argmax(output, dim=1)
            acn += torch.eq(pred, y).sum().item()
            num += x.shape[0]

            if batch_idx % 10 == 0:
                end_time = time.time()
                print(
                    f"\n[Train] Epoch: {epoch + 1}/{epochs} batch: {batch_idx}/{len(train_loader)} time: {(end_time - start_time):.1f}s Loss: {loss.item():.2f} Acc: {(100 * acn / num):.3f}%")
                acn = num = 0
                start_time = time.time()
                # print(pred)
                # print(y)
            # if batch_idx % 100 == 0:
            #     acc = eval(model, test_loader, device)
            #     print(f"\n[Evaluate] Acc: {(100 * acc):.3f}%")
        acc = eval(model, test_loader, device)
        print(f"\n[Evaluate] Acc: {(100 * acc):.3f}%")
        scheduler.step()
        
        model_name = ' Spike-DrivenV2.pkl' 
        model_name = model_name.replace(" ", "")
        torch.save(model.state_dict(), './weights/' + model_name)
        print('saved')

    # 训练完成 保存模型
    # acc = eval(model, test_loader, device)
    # acc = acc * 100
    # print(f'\n训练完成 测试集准确率: {acc:.3f}%')
    # model_name = ' Spike-DrivenV2.pkl' 
    # model_name = model_name.replace(" ", "")
    # torch.save(model.state_dict(), './weights/' + model_name)
    # print('模型已保存: ', model_name, '\n')


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


if __name__ == '__main__':
    torch.manual_seed(999)
    hp_config = [
        {'lr': 1e-3, 'batch_size': 265, 'epochs': 30, 'optim': 'Adam', 'weight_decay': 0.0005}
    ]

    for i in range(len(hp_config)):
        main(hp_config[i])