import torch
import torch.nn as nn
import torchvision.models as models


class ResNet3D(nn.Module):
    def __init__(self, num_classes):
        super(ResNet3D, self).__init__()
        self.num_classes = num_classes
        # 加载预训练的ResNet模型
        resnet = models.resnet50(pretrained=True)
        # 修改第一层为3D卷积层
        self.first_conv = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        # 使用ResNet的其余部分，除了第一层
        self.resnet = nn.Sequential(*list(resnet.children())[1:-2])
        # 将2D层替换为3D层
        self.resnet = self._transform_resnet_to_3d(self.resnet)

        self.fc = nn.Linear(2048, self.num_classes)

    def _transform_resnet_to_3d(self, resnet):
        # 创建一个新的Sequential模块
        resnet3d = nn.Sequential()
        # 遍历原始模型的所有层，并适当修改
        for name, module in resnet.named_children():
            if isinstance(module, nn.Conv2d):
                # 替换2D卷积为3D卷积
                new_module = nn.Conv3d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=(module.kernel_size[0], module.kernel_size[0], module.kernel_size[1]),
                    stride=(1, module.stride[0], module.stride[1]),
                    padding=(0, module.padding[0], module.padding[1]),
                    bias=module.bias
                )
            elif isinstance(module, nn.BatchNorm2d):
                # 替换2D批量归一化为3D批量归一化
                new_module = nn.BatchNorm3d(module.num_features)
            elif isinstance(module, nn.MaxPool2d):
                new_module = nn.MaxPool3d(
                    kernel_size=(module.kernel_size, module.kernel_size, module.kernel_size),
                    stride=(module.stride, module.stride, module.stride),
                    padding=module.padding,
                    dilation=module.dilation,
                    ceil_mode=module.ceil_mode
                )

            else:
                # 对于其他类型的层，保持不变
                new_module = module

            resnet3d.add_module(name, new_module)
        return resnet3d

    def forward(self, x):
        x = self.first_conv(x)
        x = self.resnet(x)
        x = x.mean([2, 3, 4])  # 全局平均池化
        x = self.fc(x)
        return x
