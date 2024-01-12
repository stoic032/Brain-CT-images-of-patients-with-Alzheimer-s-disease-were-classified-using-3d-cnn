import os
import h5py
import csv

import torch
from torchvision import transforms
from torch.utils.data import Dataset, random_split

# 自定义的3D数据集类，继承自torch.utils.data.Dataset
class Dataset3D(Dataset):
    def __init__(self, dataset_path, transform=None, eval_model=None):
        # 初始化数据和标签
        self.data = 0
        self.labels = []
        self.transform = transform  # 数据转换方法
        self.eval_model = eval_model  # 评估模式标志

        # 遍历数据集路径下的所有文件
        for name in os.listdir(dataset_path):
            # 如果文件是.h5格式，读取3D数据
            if name.endswith('.h5'):
                with h5py.File(os.path.join(dataset_path, name), 'r') as f:
                    self.data = f['data'][:]

            # 如果文件是.csv格式，读取标签
            elif name.endswith('.csv'):
                with open(os.path.join(dataset_path, name), 'r') as cf:
                    csv_reader = csv.reader(cf)
                    next(csv_reader, None)  # 跳过表头，从第二行开始
                    for row in csv_reader:
                        self.labels.append(int(row[1]))

    # 返回数据集的长度
    def __len__(self):
        return len(self.data)

    # 获取数据集的一个样本
    def __getitem__(self, idx):
        sample = self.data[idx]

        # 如果设置了转换方法，则应用转换
        if self.transform:
            sample = self.transform(sample)

        # 如果是评估模式，返回样本和索引
        if self.eval_model:
            return sample, idx
        else:
            # 否则返回样本和对应的标签
            label = self.labels[idx]
            return sample, label

# 自定义转换类：将4维图像转换为张量
class ToTensor4D:
    """将4维图像转换为张量。"""
    def __call__(self, x):
        # 转换为torch张量并交换维度
        # 假设x是numpy数组，形状为（1, D, H, W）
        return torch.from_numpy(x).type(torch.FloatTensor)

# 自定义转换类：归一化处理
class Normalize:
    def __call__(self, x):
        # 将数据归一化到0-1范围
        return (x - x.min()) / (x.max() - x.min())

# 定义转换流程
transform = transforms.Compose([
    ToTensor4D(),
    Normalize(),
])

# 创建训练数据集实例
# train_dataset = Dataset3D('../../data/train', transform=transform)
train_dataset = Dataset3D('./data/train', transform=transform)

# 计算数据集大小
total_size = len(train_dataset)
train_size = int(0.7 * total_size)  # 70% 作为训练集
val_size = int(0.15 * total_size)   # 15% 作为验证集
test_size = total_size - train_size - val_size  # 剩下的作为测试集

# 将数据集分割为训练集、验证集和测试集
train_dataset, val_dataset, test_dataset = random_split(train_dataset, [train_size, val_size, test_size])

# 创建评估模式的数据集实例
testa_dataset = Dataset3D('./data/test', transform=transform, eval_model=True)
testb_dataset = Dataset3D('./data/test', transform=transform, eval_model=True)



# # 小批量测试
# total_size = len(train_dataset)
# train_size = int(0.5 * total_size)  # 70% 作为训练集
# val_size = int(0.25 * total_size) # 15% 作为验证集
# test_size = total_size - train_size - val_size  # 剩下的作为测试集
# train_dataset, val_dataset, test_dataset = random_split(train_dataset, [train_size, val_size, test_size])


# #展示处理后的数据中某一层
# import matplotlib.pyplot as plt
# # 创建一个新的图像
# fig = plt.figure(figsize=(12, 9))
# for i in range(12):
#     img, label = train_dataset[i]
#     d, h, w = img[0].shape
#     # 添加子图
#     ax = fig.add_subplot(3, 4, i + 1)
#     # 显示图像
#     ax.imshow(img[0][int(d/2), :, :], 'gray')
#     # 显示标签
#     ax.text(0, h, f"Label: {label}", fontsize=12, color='red')
#
# # 调整子图之间的间距
# plt.tight_layout()
# # 显示整个图像
# plt.show()
