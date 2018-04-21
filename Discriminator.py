import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_normalization import SpectralNorm


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 定义模型参数
        # 3,256,256
        self.conv1 = SpectralNorm(nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1))
        # self.batchnorm1 = nn.BatchNorm2d(96)
        self.conv2 = SpectralNorm(nn.Conv2d(128, 256, 4, 2, 1))
        self.batchnorm2 = nn.BatchNorm2d(256)
        self.conv3 = SpectralNorm(nn.Conv2d(256, 384, 4, 2, 1))
        self.batchnorm3 = nn.BatchNorm2d(384)
        self.conv4 = SpectralNorm(nn.Conv2d(384, 384, 4, 2, 1))
        self.batchnorm4 = nn.BatchNorm2d(384)
        self.conv5 = SpectralNorm(nn.Conv2d(384, 256, 4, 2, 1))
        self.batchnorm5 = nn.BatchNorm2d(256)
        self.conv6 = SpectralNorm(nn.Conv2d(256, 128, 4, 2, 1))
        self.batchnorm6 = nn.BatchNorm2d(128)

        self.conv_last = nn.Conv2d(128, 1, 4, 4, 0)

    def forward(self, input):
        # 定义网络结构
        # 3,256,256
        conv1 = self.conv1(input)
        conv1 = F.leaky_relu(conv1)
        # conv1 = self.batchnorm1(conv1)
        # 128,128,128
        conv2 = self.conv2(conv1)
        conv2 = F.leaky_relu(conv2)
        conv2 = self.batchnorm2(conv2)
        # 256,64,64
        conv3 = self.conv3(conv2)
        conv3 = F.leaky_relu(conv3)
        conv3 = self.batchnorm3(conv3)
        # 384,32,32
        conv4 = self.conv4(conv3)
        conv4 = F.leaky_relu(conv4)
        conv4 = self.batchnorm4(conv4)
        # 384,16,16
        conv5 = self.conv5(conv4)
        conv5 = F.leaky_relu(conv5)
        conv5 = self.batchnorm5(conv5)
        # 256,8,8
        conv6 = self.conv6(conv5)
        conv6 = F.leaky_relu(conv6)
        conv6 = self.batchnorm6(conv6)
        # 128,4,4
        prob = self.conv_last(conv6)
        prob = torch.squeeze(prob)
        prob = torch.sigmoid(prob)

        return prob
