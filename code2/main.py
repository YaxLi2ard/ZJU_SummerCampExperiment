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

def main(idx):
    config_dic = hp_config[idx]
    print('\n\n参数 -> ', config_dic)
    # 部分超参数设置
    learning_rate = config_dic['lr']
    batch_size = config_dic['batch_size']
    epochs = config_dic['epochs']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据集
    train_dataset = DVS128Gesture(root='/root/autodl-tmp/DVS128/', train=True, data_type='frame', frames_number=16, split_by='number')
    test_dataset = DVS128Gesture(root='/root/autodl-tmp/DVS128/', train=False, data_type='frame', frames_number=16, split_by='number')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)

    # 创建模型
    model = Spiking_vit_MetaFormer(
        img_size_h=128,
        img_size_w=128,
        patch_size=16,
        embed_dim=[128, 256, 512, 640],
        num_heads=8,
        mlp_ratios=4,
        in_channels=2,
        num_classes=11,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=8,
        sr_ratios=1,
        drop_p=config_dic['dropout']
    )
    # model.T = 8
    checkpoint = torch.load('Spike-DrivenV2_last.pkl', map_location=device)
    model_dict = model.state_dict()
    # 只保留和模型内的参数匹配的参数权重
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model = model.to(device)
    
    # for name, param in model.named_parameters():
    #     if 'ConvBlock' in name:
    #         param.requires_grad = False

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    if config_dic['optim'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=config_dic['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.6, weight_decay=config_dic['weight_decay'])
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=15)

    # 训练
    print('---------------------------------Training---------------------------------')
    start_time = time.time()
    acn = num = 0
    max_acc = with_epoch = 0
    save_weight = None
    for epoch in range(epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.transpose(1, 0)  # [B,T,C,H,W] -> [T,B,C,H,W]
            x, y = x.to(device), y.to(device)

            model.zero_grad()
            optimizer.zero_grad()
            output = model(x)

            if config_dic['loss'] == 'CE':
                loss = criterion(output, y)
            elif config_dic['loss'] == 'MSE':
                lo = F.one_hot(y, 11).float()
                loss = F.mse_loss(output, lo)
            else:
                lo = F.one_hot(y, 11).float()
                loss = F.l1_loss(output, lo)

            loss.backward()
            optimizer.step()
            # torch.cuda.synchronize()
            functional.reset_net(model)

            # 计算分类正确的数量, 并累加样本数量
            pred = torch.argmax(output, dim=1)
            acn += torch.eq(pred, y).sum().item()
            num += y.shape[0]

            if batch_idx % 10 == 0:
                end_time = time.time()
                print(
                    f"\n[Train] Epoch: {epoch + 1}/{epochs} batch: {batch_idx}/{len(train_loader)} time: {(end_time - start_time):.1f}s Loss: {loss.item():.2f} Acc: {(100 * acn / num):.3f}%")
                acn = num = 0
                start_time = time.time()
                # print(pred)
                # print(y)
        acc = eval(model, test_loader, device)
        print(f"\n[Evaluate] Acc: {(100 * acc):.3f}%")
        # if acc >= max_acc:
        #     max_acc = acc
        #     with_epoch = epoch + 1
        #     save_weight = model.state_dict()
        acc = acc * 100
        model_name = f'{idx}_Spike-DrivenV2_{epoch + 1}_' + str(config_dic) + f'_{acc:.3f}_' + '.pkl'
        model_name = model_name.replace(" ", "")
        torch.save(model.state_dict(), './weights/' + model_name)
        print('saved')
        scheduler.step()

    # 训练完成 保存模型
    # max_acc = max_acc * 100
    # print(f'\n训练完成 测试集最高准确率: {max_acc:.3f}%')
    # model_name = f'{idx}_Spike-DrivenV2_{with_epoch}_' + str(config_dic) + f'_{max_acc:.3f}_' + '.pkl'
    # model_name = model_name.replace(" ", "")
    # torch.save(save_weight, './weights/' + model_name)
    # print('模型已保存: ', model_name, '\n')


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


if __name__ == '__main__':
    torch.manual_seed(999)
    hp_config = [
        {},
        # 1
        {'lr': 1e-4, 'batch_size': 6, 'epochs': 30, 'loss': 'CE', 'dropout': 0.1, 'optim': 'Adam', 'weight_decay': 0.000},
        # 2
        {'lr': 1e-4, 'batch_size': 6, 'epochs': 30, 'loss': 'CE', 'dropout': 0.3, 'optim': 'Adam', 'weight_decay': 0.0005},
        # 3
        {'lr': 1e-4, 'batch_size': 6, 'epochs': 30, 'loss': 'CE', 'dropout': 0.5, 'optim': 'Adam', 'weight_decay': 0.0005},
        # 4
        {'lr': 1e-4, 'batch_size': 5, 'epochs': 30, 'loss': 'CE', 'dropout': 0.1, 'optim': 'Adam', 'weight_decay': 0.0005},
        # 5
        {'lr': 1e-4, 'batch_size': 9, 'epochs': 30, 'loss': 'CE', 'dropout': 0.3, 'optim': 'Adam', 'weight_decay': 0.0005},
        # 6
        {'lr': 1e-4, 'batch_size': 9, 'epochs': 30, 'loss': 'CE', 'dropout': 0.5, 'optim': 'Adam', 'weight_decay': 0.0005},
        # 7
        {'lr': 1e-4, 'batch_size': 9, 'epochs': 30, 'loss': 'CE', 'dropout': 0.3, 'optim': 'Adam', 'weight_decay': 0.005},
        # 8
        {'lr': 1e-4, 'batch_size': 9, 'epochs': 30, 'loss': 'CE', 'dropout': 0.6, 'optim': 'Adam', 'weight_decay': 0.0005},
    ]

    # for i in range(len(hp_config)):
    #     if i == 0: continue
    #     main(i)
    main(3)