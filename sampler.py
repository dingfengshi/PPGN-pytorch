import torch
import ppgn
import modified_alexnet
import DataUtill
import torch.optim as optim
import torch.utils.data as datas
import torch.nn as nn
from torch.autograd import *

n_iter = 10
data_path = "place365_standard/"
epsilon1 = 1e-5
epsilon2 = -1
epsilon3 = 1e-17


def sample(condition_class, samples=20, sample_step=10):
    D = ppgn.Discriminator()
    G = ppgn.Generator()
    encoder = modified_alexnet()
    classifier = torch.load("whole_alexnet_places365_python36.pth.tar")
    modified_alexnet.get_weight(encoder)

    D.cuda()
    G.cuda()
    encoder.cuda()
    classifier.cuda()

    class_loss = nn.NLLLoss()
    condition_class = Variable(torch.FloatTensor(condition_class))

    trans = DataUtill.get_ImageNet_transform((256, 256), imagenet_normalize=False)
    train_data = DataUtill.Placesdataset(data_path, transforms=trans)
    train_data_loader = datas.DataLoader(train_data, 1, shuffle=True, num_workers=1)

    idx = 0
    while True:
        for _, batch in enumerate(train_data_loader):
            # 得到初始h
            img = batch["img"]
            pool5, fc6 = ppgn.get_encode(encoder, img)
            h = Variable(fc6, requires_grad=True).cuda()
            e2optim = optim.SGD(h, epsilon2)

            for i in range(sample_step):
                # 1.计算epsilon1项
                _, r_h = encoder(G(h))
                e1_step = epsilon1 * (r_h - h)

                # 2.计算epsilon2项
                G.zero_grad()
                D.zero_grad()
                gen = G(h)
                output = classifier(gen)
                loss = class_loss(output, condition_class)
                loss.backward()
                e2optim.step()

                # 3.计算epsilon3项
                h=h+e1_step+torch.normal()

            idx = idx + 1
            if idx >= samples:
                break
