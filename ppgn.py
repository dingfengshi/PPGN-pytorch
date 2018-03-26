import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as datas
import torchvision
import torchvision.transforms as transform
from tensorboardX import SummaryWriter
from torch.autograd import Variable

import DataUtill
import modified_alexnet

# 学习参数
d_learning_rate = 2e-4
g_learning_rate = 2e-4
optim_betas = (0.5, 0.999)
num_epochs = 200
lambda_feature = 1
lambda_pixel = 1
real_smooth = 0.9
fake_smooth = 0.1

# 训练比例
d_steps = 5
g_steps = 1
model_path = "model/"
data_path = "/dev/shm/places365_standard/"
batch_size = 64


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 定义模型参数
        self.defc7 = nn.Linear(4096, 4096)
        self.defc6 = nn.Linear(4096, 4096)
        self.defc5 = nn.Linear(4096, 4096)
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
        x = self.defc6(x)
        x = F.leaky_relu(x)
        x = self.defc5(x)
        x = F.leaky_relu(x)
        x = x.view(-1, 256, 4, 4)
        # upsampling
        deconv5 = self.deconv5(x)
        deconv5 = F.leaky_relu(deconv5)
        # (256,8,8)
        conv5_1 = self.conv5_1(deconv5)
        conv5_1 = F.leaky_relu(conv5_1)
        # (512,8,8)
        deconv4 = self.deconv4(conv5_1)
        deconv4 = F.leaky_relu(deconv4)
        # (256,16,16)
        conv4_1 = self.conv4_1(deconv4)
        conv4_1 = F.leaky_relu(conv4_1)
        # (256,16,16)
        deconv3 = self.deconv3(conv4_1)
        deconv3 = F.leaky_relu(deconv3)
        # (128,32,32)
        conv3_1 = self.conv3_1(deconv3)
        conv3_1 = F.leaky_relu(conv3_1)
        # (128,32,32)
        deconv2 = self.deconv2(conv3_1)
        deconv2 = F.leaky_relu(deconv2)
        # (64,64,64)
        deconv1 = self.deconv1(deconv2)
        deconv1 = F.leaky_relu(deconv1)
        # (32,128,128)
        deconv0 = self.deconv0(deconv1)
        gen = F.tanh(deconv0)
        # (3,256,256)
        return gen


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 定义模型参数
        # 3,256,256
        self.conv1 = nn.Conv2d(3, 96, kernel_size=4, stride=2, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 256, 4, 2, 1)
        self.batchnorm2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 384, 4, 2, 1)
        self.batchnorm3 = nn.BatchNorm2d(384)
        self.conv4 = nn.Conv2d(384, 384, 4, 2, 1)
        self.batchnorm4 = nn.BatchNorm2d(384)
        self.conv5 = nn.Conv2d(384, 256, 4, 2, 1)
        self.batchnorm5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 1, 8, 8, 0)

    def forward(self, input):
        # 定义网络结构
        # 3,256,256
        conv1 = self.conv1(input)
        conv1 = F.leaky_relu(conv1)
        conv1 = self.batchnorm1(conv1)
        # 96,128,128
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
        prob = self.conv6(conv5)
        prob = torch.squeeze(prob)
        prob = torch.sigmoid(prob)

        return prob


def weight_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        init.normal(m.weight.data, 0, 0.002)
        init.constant(m.bias.data, 0)


def get_encode(encoder, img):
    img = img[:, :, 16:240, 16:240].data
    img = img * 0.5 + 0.5
    trans = transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    imgs = []
    for i in range(img.size(0)):
        imgs.append(trans(img[i]).unsqueeze(0))
    img = torch.cat(imgs, 0)
    img = Variable(img).cuda()
    return encoder(img)


def train():
    G = Generator()
    D = Discriminator()
    encoder = modified_alexnet.modifiedAlexNet()
    # 导入encoder的预训练权重
    modified_alexnet.get_weight(encoder)

    # fc6:4096,pool5:256*6*6

    for param in list(encoder.parameters()):
        param.requires_grad = False

    G.apply(weight_init)
    D.apply(weight_init)
    G.train()
    D.train()

    global_step, now_batch = DataUtill.restore_checkpoint({"generator": G, "discriminator": D}, model_path)

    encoder.cuda()
    G.cuda()
    D.cuda()
    # 定义损失
    # 对抗损失
    adv_criterion = nn.BCELoss()
    # image损失
    pixel_criterion = nn.MSELoss()
    d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate, momentum=0.9, weight_decay=0.0004)
    g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas, weight_decay=0.0004)

    d_optimizer.param_groups[0]["initial_lr"] = d_learning_rate
    g_optimizer.param_groups[0]["initial_lr"] = g_learning_rate

    d_optimizer = optim.lr_scheduler.ExponentialLR(d_optimizer, 0.5, now_batch)
    g_optimizer = optim.lr_scheduler.ExponentialLR(g_optimizer, 0.5, now_batch)

    # d_optimizer = optim.lr_scheduler.ExponentialLR(d_optimizer, 0.5, now_batch)

    # log目录
    writer = SummaryWriter(log_dir=model_path)

    # 取得数据集
    trans = DataUtill.get_ImageNet_transform((256, 256), imagenet_normalize=False, ordinary_normalize=True)
    train_data = DataUtill.Placesdataset(data_path, transforms=trans)
    train_data_loader = datas.DataLoader(train_data, batch_size, shuffle=True, num_workers=10)

    for name, param in D.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step)
    for name, param in G.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step)

    for epoch in range(now_batch, num_epochs):
        for step, batch_data in enumerate(train_data_loader):
            # 训练判别器
            global_step = global_step + 1
            for d_index in range(d_steps):
                D.zero_grad()

                # 训练真实图像
                d_real_data = Variable(batch_data["img"]).cuda()
                d_real_decision = D(d_real_data)
                d_real_error = adv_criterion(d_real_decision,
                                             Variable(real_smooth * torch.ones(d_real_data.size(0)).cuda()))
                d_real_error.backward()
                # test
                # torchvision.utils.save_image(d_real_data.data, "real.jpg", normalize=True)

                # 训练生成图像
                real_pool5, d_gen_input = get_encode(encoder, d_real_data)
                d_gen_input = Variable(d_gen_input)
                d_fake_data = G(d_gen_input).detach()  # 中断梯度
                # test
                # torchvision.utils.save_image(d_fake_data.data, "gen1.jpg", normalize=True)

                d_fake_decision = D(d_fake_data)
                d_fake_error = adv_criterion(d_fake_decision,
                                             Variable(fake_smooth + torch.zeros(d_fake_data.size(0)).cuda()))
                d_fake_error.backward()
                d_optimizer.step()

            for g_index in range(g_steps):
                # 训练generator
                G.zero_grad()

                # 1.pixel loss+feature_loss
                # g_real_data = Variable(batch_data["img"]).cuda()
                # real_pool5, fc6 = get_encode(encoder, d_real_data)
                # gen_input = Variable(d_gen_input)
                g_fake_data = G(d_gen_input)

                # test
                # torchvision.utils.save_image(g_fake_data.data, "gen2.jpg")

                pixel_loss = lambda_pixel * pixel_criterion(g_fake_data, d_real_data)

                gen_pool5, fc6 = get_encode(encoder, g_fake_data)
                feature_loss = torch.mean(torch.pow((gen_pool5 - real_pool5), 2)) * lambda_feature

                pixel_loss.data = pixel_loss.data + feature_loss

                pixel_loss.backward(retain_graph=True)

                # 2.对抗损失
                dg_fake_decision = D(g_fake_data)
                g_error = adv_criterion(dg_fake_decision,
                                        Variable(torch.ones(g_fake_data.size(0)).cuda()))
                g_error.backward()

                g_optimizer.step()

            if global_step % 200 == 0:
                writer.add_scalars("/loss/Discriminator_loss", {'real_loss': d_real_error.data[0],
                                                                'fake_loss': d_fake_error.data[0]}, global_step)
                writer.add_scalar("/loss/pixel_loss", pixel_loss.data[0], global_step)
                writer.add_scalar("/loss/adv_loss", g_error.data[0], global_step)
                writer.add_scalar("/loss/feature_loss", feature_loss, global_step)
                # writer.add_scalar("/loss/total_loss", pixel_loss.data[0] + g_error.data[0], global_step)
                # for name, param in D.named_parameters():
                #     writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step)
                # writer.add_scalar("learning_rate", DataUtill.get_learning_rate_from_optim(g_optimizer), global_step)

            if global_step % 2000 == 0:
                torchvision.utils.save_image(g_fake_data.data, "output/" + str(global_step) + ".jpg", normalize=True)
                DataUtill.save_checkpoint({
                    "generator": G,
                    "discriminator": D
                }, global_step, epoch, model_path)
                writer.add_scalar("Discriminator_learning_rate",
                                  DataUtill.get_learning_rate_from_scheduler_optim(d_optimizer), global_step)
        print("epoch {} finished".format(epoch))

    writer.close()


if __name__ == '__main__':
    train()
