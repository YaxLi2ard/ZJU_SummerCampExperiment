import numpy as np
import torch
import linecache
import torch.utils.data as data
import os

# 打开原始文件
with open('./ES-imagenet-0.18/trainlabel.txt', 'r') as infile:
    lines = infile.readlines()

# 处理每一行
processed_lines = []
for line in lines:
    filename, classnum, a, b = line.split()
    filename = filename[:-3] + 'npz'
    file_path = './ES-imagenet-0.18/train/' + filename
    if os.path.exists(file_path):
        processed_lines.append(line)

# 将结果写入新文件
with open('processed_trainlabel.txt', 'w') as outfile:
    outfile.writelines(processed_lines)