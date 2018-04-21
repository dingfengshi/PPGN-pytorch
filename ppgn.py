import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as datas
import torchvision
import torchvision.transforms as transform
from tensorboardX import SummaryWriter
from torch.autograd import Variable

import DataUtill
import modified_alexnet
from Discriminator import Discriminator
from Generator import Generator

# 学习参数
d_learning_rate = 4e-4
g_learning_rate = 4e-4
optim_betas = (0.5, 0.999)
num_epochs = 1000
lambda_feature = 1
lambda_pixel = 1
real_smooth = 0.9
fake_smooth = 0
orth_reg = 1e-6

# 训练比例
d_steps = 1
g_steps = 1
model_path = "model/"
data_path = "/home/ste/places365_standard/"
batch_size = 64
save_step = 2000


def weight_init(m):
    if isinstance(m, nn.ConvTranspose2d):
        init.orthogonal(m.weight.data, 1.0)
    elif isinstance(m, nn.Conv2d):
        if hasattr(m, "weight"):
            init.orthogonal(m.weight.data, 1.0)
        else:
            init.orthogonal(m.weight_bar.data, 1.0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0)


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

    encoder.cuda()
    G.cuda()
    D.cuda()

    d_optimizer = optim.SGD([i for i in D.parameters() if i.requires_grad == True], lr=d_learning_rate, momentum=0.9)
    g_optimizer = optim.Adam([i for i in G.parameters() if i.requires_grad == True], lr=g_learning_rate,
                             betas=optim_betas)

    global_step, now_batch = DataUtill.restore_checkpoint({"generator": G,
                                                           "discriminator": D,
                                                           "d_optimizer": d_optimizer,
                                                           "g_optimizer": g_optimizer
                                                           }, model_path)

    # d_optimizer.param_groups[0]["initial_lr"] = d_learning_rate
    # g_optimizer.param_groups[0]["initial_lr"] = g_learning_rate
    d_optimizer.param_groups[0]["lr"] = d_learning_rate
    g_optimizer.param_groups[0]["lr"] = g_learning_rate

    G.train()
    D.train()

    # 定义损失
    # 对抗损失
    adv_criterion = nn.BCELoss()
    # image损失
    pixel_criterion = nn.MSELoss()

    # d_lr_optim = optim.lr_scheduler.ExponentialLR(d_optimizer, 0.998, now_batch)
    # g_lr_optim = optim.lr_scheduler.ExponentialLR(g_optimizer, 0.998, now_batch)

    # log目录
    writer = SummaryWriter(log_dir=model_path)

    # 取得数据集
    trans = DataUtill.get_ImageNet_transform((256, 256), imagenet_normalize=False, ordinary_normalize=True)
    train_data = DataUtill.Placesdataset(data_path, transforms=trans)
    train_data_loader = datas.DataLoader(train_data, batch_size, shuffle=True, num_workers=16)

    for epoch in range(now_batch, num_epochs):
        for step, batch_data in enumerate(train_data_loader):
            global_step = global_step + 1
            d_real_data = Variable(batch_data["img"]).cuda()
            real_pool5, d_gen_input = get_encode(encoder, d_real_data)
            d_gen_input = Variable(d_gen_input)

            # 训练判别器
            for d_index in range(d_steps):
                D.zero_grad()

                # 训练真实图像
                d_real_decision = D(d_real_data)
                d_real_error = adv_criterion(d_real_decision,
                                             Variable(real_smooth * torch.ones(d_real_data.size(0)).cuda()))
                d_real_error.backward()

                d_fake_data = G(d_gen_input).detach()  # 中断梯度
                # 训练生成图像
                d_fake_decision = D(d_fake_data)
                d_fake_error = adv_criterion(d_fake_decision,
                                             Variable(fake_smooth + torch.zeros(d_fake_data.size(0)).cuda()))
                d_fake_error.backward()

                orth_loss = Variable(torch.FloatTensor(1), requires_grad=True).cuda()

                for name, param in D.named_parameters():
                    if 'bias' not in name:
                        param_flat = param.view(param.shape[0], -1)
                        sym = torch.mm(param_flat, torch.t(param_flat))
                        sym -= Variable(torch.eye(param_flat.shape[0])).cuda()
                        orth_loss = orth_loss + (orth_reg * sym.sum())
                orth_loss.backward()

                d_optimizer.step()

            for g_index in range(g_steps):
                # 训练generator
                G.zero_grad()

                # 1.pixel loss+feature_loss
                g_fake_data = G(d_gen_input)

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

            if global_step % (save_step // 5) == 0:
                if d_steps > 0:
                    writer.add_scalars("/loss/Discriminator_loss", {'real_loss': d_real_error.data[0],
                                                                    'fake_loss': d_fake_error.data[0]}, global_step)
                    writer.add_scalar("/loss/feature_loss", feature_loss, global_step)

                writer.add_scalar("/loss/pixel_loss", pixel_loss.data[0], global_step)
                writer.add_scalar("/loss/adv_loss", g_error.data[0], global_step)
                writer.add_scalar("/loss/total_loss", pixel_loss.data[0] + g_error.data[0], global_step)
                # for name, param in D.named_parameters():
                #     writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step)
                # writer.add_scalar("learning_rate", DataUtill.get_learning_rate_from_optim(g_optimizer), global_step)

            if global_step % save_step == 0:
                torchvision.utils.save_image(g_fake_data.data, "output/" + str(global_step) + ".jpg", normalize=True)
                DataUtill.save_checkpoint({
                    "generator": G,
                    "discriminator": D,
                    "d_optimizer": d_optimizer,
                    "g_optimizer": g_optimizer
                }, global_step, epoch, model_path)
                writer.add_scalar("Discriminator_learning_rate",
                                  DataUtill.get_learning_rate_from_optim(d_optimizer), global_step)
                writer.add_scalar("Generator_learning_rate",
                                  DataUtill.get_learning_rate_from_optim(g_optimizer), global_step)
            if global_step % (save_step * 3) == 0:
                DataUtill.save_checkpoint({
                    "generator_" + str(global_step): G,
                    "discriminator_" + str(global_step): D
                }, global_step, epoch, model_path)

        # g_lr_optim.step()
        # d_lr_optim.step()
        print("epoch {} finished".format(epoch))

    writer.close()


if __name__ == '__main__':
    train()
