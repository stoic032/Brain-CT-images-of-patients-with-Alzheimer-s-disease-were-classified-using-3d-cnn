import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class cnn3D(nn.Module):
    def __init__(self, num_classes, input_size=(79, 95, 79)):
        super(cnn3D, self).__init__()
        # 正常样本、轻度认知障碍样本、阿尔茨海默症样本
        self.num_classes = num_classes

        # 对于阿尔茨海默病样本数据结构为：(79,95,79),看成连续79张(95, 79)图片
        self.depth, self.height, self.width = input_size

        # 定义用于计算梯度特征的卷积层
        self.gradient_conv_x = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.gradient_conv_y = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=1, padding=1, bias=False)

        # 初始化梯度卷积核
        with torch.no_grad():
            self.gradient_conv_x.weight = nn.Parameter(torch.tensor([[[[-1, 0, 1],
                                                                        [-2, 0, 2],
                                                                        [-1, 0, 1]]]], dtype=torch.float32))
            self.gradient_conv_y.weight = nn.Parameter(torch.tensor([[[[-1, -2, -1],
                                                                        [0, 0, 0],
                                                                        [1, 2, 1]]]], dtype=torch.float32))

        # C2层：用两种不同的3D卷积核的7*7*11的3D卷积核对5个channels分别进行卷积，获得两个系列
        # 2*3*77@73*89
        self.conv2_1= nn.Conv3d(in_channels=1 * 3, out_channels=3, kernel_size=(7, 7, 3), stride=1, padding=0)
        self.conv2_2= nn.Conv3d(in_channels=1 * 3, out_channels=3, kernel_size=(7, 7, 3), stride=1, padding=0)

        # S3层：2x2池化，下采样
        # 2*3*39@37*45
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=3)

        # C4层：用三个7*7*3的3D卷积核分别对各个系列各个channels进行卷积，获得6个系列
        # 2*3*3*37@31*39
        self.conv4 = nn.Conv3d(in_channels=3, out_channels=3 * 3, kernel_size=(7, 7, 3), stride=1, padding=0)

        # S5层：2X2池化，下采样
        # 2*3*3*19@16*20
        self.maxpool5 = nn.MaxPool3d(kernel_size=2, padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=9)

        # C6层：用三个7*7*3的3D卷积核分别对各个系列各个channels进行卷积，获得18个系列
        # 2*3*3*3*17@10*14
        self.conv6 = nn.Conv3d(in_channels=3 * 3, out_channels=9 * 3, kernel_size=(7, 7, 3), stride=1, padding=0)

        # S7层：2X2池化，下采样
        # 2*3*3*3*9@6*8
        self.maxpool7 = nn.MaxPool3d(kernel_size=2, padding=1)
        self.bn3 = nn.BatchNorm3d(num_features=27)

        # C8层：用三个7*7*3的3D卷积核分别对各个系列各个channels进行卷积，获得54个系列
        # 2*3*3*3*3*3@4*6
        self.conv8 = nn.Conv3d(in_channels=9 * 3, out_channels=27 * 3, kernel_size=(3, 3, 7), stride=1, padding=0)

        # S9层
        # 2*3*3*3*3*3*1@2*3
        self.avgpool9 = nn.AvgPool3d(kernel_size=3, padding=1)

        self.flatten = nn.Flatten()

        self.fc10 = nn.Linear(648, 256)
        self.dropout = nn.Dropout(p=0.2)
        self.fc11 = nn.Linear(256, self.num_classes)

    def gray(self, x):
        """如果为多通道图像，就导出灰度"""
        pass

    def feature(self, x):
        x = F.relu(x)
        x = self.maxpool3(x)
        x = self.bn1(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.maxpool5(x)
        x = self.bn2(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.maxpool7(x)
        x = self.bn3(x)
        x = self.conv8(x)
        x = F.relu(x)

        return x

    def forward(self, x):
        batch_size, channels, depth, height, width = x.shape

        x_2d = x.view(batch_size * depth, channels, height, width)

        # 应用2D卷积
        grad_x_2d = self.gradient_conv_x(x_2d)
        grad_y_2d = self.gradient_conv_y(x_2d)

        # 将结果重新变形为3D格式
        grad_x = grad_x_2d.view(batch_size, channels, depth, height, width)
        grad_y = grad_y_2d.view(batch_size, channels, depth, height, width)

        x = torch.cat([x, grad_x, grad_y], dim=1)       # 在第2个通道合并

        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)

        x1 = self.feature(x1)
        x2 = self.feature(x2)

        # 合并特征map
        x = torch.cat((x1, x2), dim=1)      # 在第2个通道合并

        x = self.avgpool9(x)
        x = self.flatten(x)
        x = self.fc10(x)
        x = self.dropout(x)
        x = self.fc11(x)

        return x
