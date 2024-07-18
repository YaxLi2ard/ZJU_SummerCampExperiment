import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ProcessedDataset(Dataset):
    def __init__(self, root_dir):
        """
        初始化数据集，读取指定目录中的所有子文件夹中的所有npz文件。

        参数:
        root_dir (str): 根目录路径。
        """
        self.root_dir = root_dir
        self.file_paths = []

        # 遍历根目录及其子文件夹，收集所有的npz文件路径
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.npz'):
                    self.file_paths.append(os.path.join(subdir, file))

    def __len__(self):
        """
        返回数据集中的样本数量。
        """
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        根据索引获取数据集中的样本。

        参数:
        idx (int): 样本索引。

        返回:
        tuple: 包含输入x和标签y的元组。
        """
        file_path = self.file_paths[idx]
        data = np.load(file_path)

        x = data['x']
        y = data['y']

        # 将numpy数组转换为PyTorch张量
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y


# 示例用法
if __name__ == "__main__":
    root_dir = './processed/'  # 替换为你的根目录路径
    dataset = ProcessedDataset(root_dir)

    # 测试数据集
    print(f"数据集中样本数量: {len(dataset)}")
    x, y = dataset[0]
    print(x.shape)
    print(y)
