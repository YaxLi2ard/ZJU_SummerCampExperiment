import torch
import torch.nn as nn
import torch.optim as optim
import sys
import torch.nn.functional as F
from torch.cuda import amp
from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net
from processed_dataset import ProcessedDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import os
import argparse
import datetime
from LIAF import *
from LIAFResNet import *
from LIAFResNet_18 import Config

def main():
    # 部分超参数设置
    learning_rate = 0.0001
    batch_size = 50
    epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据集
    train_dataset = ProcessedDataset(root_dir='./processed/')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)

    # 创建模型
    config = Config()
    model = LIAFResNet(config)
    
    checkpoint = torch.load('ResNet18BasedOnSNN_copy_last.pkl', map_location=device)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0000)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=30)

    # 训练
    print('---------------------------------Training---------------------------------')
    acc_list, loss_list = [], []
    start_time = time.time()
    acn = num = loss_sum = loss_num = 0
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
            num += y.shape[0]
            loss_sum += loss.item()
            loss_num += 1

            if batch_idx % 10 == 0:
                acc_list.append(acn / num)
                loss_list.append(loss_sum / loss_num)
                end_time = time.time()
                print(
                    f"\n[Train] Epoch: {epoch + 1}/{epochs} batch: {batch_idx}/{len(train_loader)} time: {(end_time - start_time):.1f}s Loss: {(loss_sum / loss_num):.2f} Acc: {(100 * acn / num):.3f}%")
                acn = num = loss_sum = loss_num = 0
                start_time = time.time()
                print(pred[:6])
                print(y[:6])
            if batch_idx % 200 == 0 and batch_idx != 0:
                torch.save(model.state_dict(), 'ResNet18BasedOnSNN_last.pkl')
                acc_np = np.array(acc_list)
                loss_np = np.array(loss_list)
                np.save('acc_list2.npy', acc_np)
                np.save('loss_list2.npy', loss_np)
                print('saved')
    
        # acc = eval(model, test_loader, device)
        # print(f"\n[Evaluate] Acc: {(100 * acc):.3f}%")
        scheduler.step()

    # 训练完成 保存模型
    acc = eval(model, test_loader, device)
    acc = acc * 100
    print(f'\n训练完成 测试集准确率: {acc:.3f}%')
    model_name = 'resnet_' + str(config_dic) + f'_{acc:.3f}_' + '.pkl'
    model_name = model_name.replace(" ", "")
    torch.save(model.state_dict(), './weights/' + model_name)
    print('模型已保存: ', model_name, '\n')


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
            num += y.shape[0]
    return acn / num


if __name__ == '__main__':
    torch.manual_seed(999)
    main()