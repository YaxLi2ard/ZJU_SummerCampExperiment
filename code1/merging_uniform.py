import os
import torch
import glob
import pickle
from collections import OrderedDict
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torch.utils.data import DataLoader
from LIAF import *
from LIAFResNet import *
from LIAFResNet_18 import Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_dataset = DVS128Gesture(root='/root/autodl-tmp/DVS128/', train=False, data_type='frame', frames_number=16, split_by='number')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
    
def load_model_weights(file_path):
    return torch.load(file_path, map_location=device)

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
            num += x.shape[0]
    return acn / num

def get_model_from_sd(state_dict):
    config = Config()
    model = LIAFResNet(config)
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
    accuracy = float(parts[-2])  # 假设倒数第二部分是准确率
    models_with_acc.append((file_path, accuracy))

# 按准确率从高到低排序模型
models_with_acc.sort(key=lambda x: x[1], reverse=True)
models_with_acc = models_with_acc[:]

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

torch.save(model.state_dict(), './resnet_uniformsoup.pkl')
print('合并模型已保存')