import numpy as np
import os
import torch
import linecache
import torch.utils.data as data
from tqdm import tqdm


class ESImagenet_Dataset(data.Dataset):
    def __init__(self, mode, data_set_path='/data/dvsimagenet/'):
        super().__init__()
        self.mode = mode
        self.filenames = []
        self.trainpath = data_set_path + 'train'
        self.testpath = data_set_path + 'val'
        self.traininfotxt = data_set_path + 'trainlabel.txt'
        self.testinfotxt = data_set_path + 'vallabel.txt'
        self.formats = '.npz'
        if mode == 'train':
            self.path = self.trainpath
            trainfile = open(self.traininfotxt, 'r')
            for line in trainfile:
                filename, classnum, a, b = line.split()
                realname, sub = filename.split('.')
                self.filenames.append(realname + self.formats)
        else:
            self.path = self.testpath
            testfile = open(self.testinfotxt, 'r')
            for line in testfile:
                filename, classnum, a, b = line.split()
                realname, sub = filename.split('.')
                self.filenames.append(realname + self.formats)

    def __getitem__(self, index):
        if self.mode == 'train':
            info = linecache.getline(self.traininfotxt, index + 1)
        else:
            info = linecache.getline(self.testinfotxt, index + 1)
        filename, classnum, a, b = info.split()
        realname, sub = filename.split('.')
        filename = realname + self.formats
        filename = self.path + r'/' + filename
        classnum = int(classnum)
        a = int(a)
        b = int(b)
        datapos = np.load(filename)['pos'].astype(np.float64)
        dataneg = np.load(filename)['neg'].astype(np.float64)

        dy = (254 - b) // 2
        dx = (254 - a) // 2
        input = torch.zeros([2, 8, 256, 256])

        x = datapos[:, 0]
        y = datapos[:, 1]
        t = datapos[:, 2] - 1
        input[0, t, x, y] = 1

        x = dataneg[:, 0]
        y = dataneg[:, 1]
        t = dataneg[:, 2] - 1
        input[0, t, x, y] = 1

        reshape = input[:, :, 16:240, 16:240]
        label = torch.tensor([classnum])
        return reshape, label

    def __len__(self):
        return len(self.filenames)


def read_and_map_b(file_path):
    """
    读取txt文件中的每一行，统计出现的b值并按b值的大小排序，
    对b值与其排序后的下标进行映射。

    参数:
    file_path (str): txt文件的路径。

    返回:
    dict: b值到其排序后下标的映射字典。
    """
    b_values = []

    # 读取文件并提取每行的b值
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 4:
                b = int(parts[1])  # 提取b值并转换为整数
                b_values.append(b)

    # 获取唯一的b值并排序
    unique_b_values = sorted(set(b_values))

    # 创建b值到下标的映射
    b_to_index = {b: index for index, b in enumerate(unique_b_values)}

    return b_to_index


# for i in range(101):
#     os.makedirs(f'./processed/{i}')

data_path = './ES-imagenet-0.18/'
train_dataset = ESImagenet_Dataset(mode='train', data_set_path=data_path)

dic = read_and_map_b('./ES-imagenet-0.18/trainlabel.txt')

cnt = [0 for i in range(len(train_dataset) + 9)]
pbar = tqdm(total=len(train_dataset))
for i in range(len(train_dataset)):
    x, y = train_dataset[i]
    x, y = x.numpy(), y.numpy()
    label = y[0]
    label_ = dic[label]
    idx = cnt[label_]
    cnt[label_] += 1
    np.savez_compressed(f'./processed/{label_}/{label_}_{label}_{idx}.npz', x=x, y=np.array(label_))
    if (i + 1) % 10 == 0:
        pbar.update(10)
pbar.close()

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=60, shuffle=False, num_workers=1,pin_memory=True,drop_last=True)

# for batch_idx, (inputs, targets) in enumerate(train_loader):
#     print(f'{batch_idx}/{len(train_loader)}', inputs.shape)

# data = np.load('./processed/36/36_135_0.npz')
# x = data['x']
# y = data['y']
# print(x.shape)
# print(y)