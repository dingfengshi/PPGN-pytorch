import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 定义模型参数
        self.defc7 = nn.Linear(4096, 4096)
        # self.defc6 = nn.Linear(4096, 4096)
        # self.defc5 = nn.Linear(4096, 4096)
        # s'=(s-1)*stride-2p+kernel
        self.deconv5 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.conv5_1 = nn.ConvTranspose2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.deconv4 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.conv4_1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv3_1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv0 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, feature):
        # 定义网络结构
        # 4096
        x = self.defc7(feature)
        x = F.leaky_relu(x)
        # x = self.defc6(x)
        # x = F.leaky_relu(x)
        # x = self.defc5(x)
        # x = F.leaky_relu(x)
        x = x.view(-1, 256, 4, 4)
        deconv5 = self.deconv5(x)
        deconv5 = F.leaky_relu(deconv5)
        # (256,8,8)
        conv5_1 = self.conv5_1(deconv5)
        conv5_1 = F.leaky_relu(conv5_1)
        # (512,8,8)
        deconv4 = self.deconv4(conv5_1)
        deconv4 = F.leaky_relu(deconv4)
        deconv4 = self.conv4_1(deconv4)
        # (256,16,16)
        deconv3 = self.deconv3(deconv4)
        deconv3 = F.leaky_relu(deconv3)
        deconv3 = self.conv3_1(deconv3)
        # (128,32,32)
        deconv2 = self.deconv2(deconv3)
        deconv2 = F.leaky_relu(deconv2)
        # (64,64,64)
        deconv1 = self.deconv1(deconv2)
        deconv1 = F.leaky_relu(deconv1)
        # (32,128,128)
        deconv0 = self.deconv0(deconv1)
        gen = F.tanh(deconv0)
        # (3,256,256)
        return gen
