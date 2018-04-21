import torch
import ppgn
import modified_alexnet
import DataUtill
import torch.optim as optim
import torch.utils.data as datas
import torch.nn as nn
from torch.autograd import *
import torchvision.transforms as transform
import torchvision
import numpy as np

n_iter = 10
data_path = "/dev/shm/places365_standard/"
model_path = "model/"
epsilon1 = 1e-5
epsilon2 = -1
epsilon3 = 1e-34

# means = Variable(torch.FloatTensor([0.485, 0.456, 0.406])).cuda()
# std = Variable(torch.FloatTensor([0.229, 0.224, 0.225])).cuda()
means = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def alex_transfrom(img):
    img_norm = img.squeeze() * 0.5 + 0.5
    c_out = []
    for c, m, s in zip(img_norm, means, std):
        c_out.append((c - m) / s)
    img_norm = torch.cat(c_out, 0)
    return img_norm.view([1, 3, 256, 256])[:, :, 16:240, 16:240]


def sample(samples=10):
    G = ppgn.Generator()
    encoder = modified_alexnet.modifiedAlexNet()
    classifier = torch.load("whole_alexnet_places365_python36.pth.tar")
    modified_alexnet.get_weight(encoder)
    DataUtill.restore_checkpoint({"generator": G}, model_path)

    G.cuda()
    encoder.cuda()
    classifier.cuda()

    class_loss = nn.NLLLoss()

    trans = DataUtill.get_ImageNet_transform((256, 256), imagenet_normalize=False)
    train_data = DataUtill.Placesdataset(data_path, transforms=trans)
    train_data_loader = datas.DataLoader(train_data, 1, shuffle=True, num_workers=1)

    idx = 0
    while True:
        for pic_num, batch in enumerate(train_data_loader):
            condition_class = Variable((batch["class"]).squeeze()).cuda()
            # 得到初始h
            img = Variable(batch["img"]).cuda()
            pool5, fc6 = ppgn.get_encode(encoder, img)
            h = Variable(fc6.cuda(), requires_grad=True)
            e2optim = optim.SGD([h], epsilon2)

            for i in range(n_iter):
                # 1.计算epsilon1项
                gen = G(h)
                torchvision.utils.save_image(gen.data, "sample/test{}.jpg".format(pic_num), 1, normalize=True)
                alex_gen = alex_transfrom(gen)
                _, r_h = encoder(alex_gen)
                e1_step = epsilon1 * (r_h - h.data)
                h.data = h.data + e1_step

                # 2.计算epsilon2项
                h.grad = None
                G.zero_grad()
                output = classifier(alex_gen)
                loss = class_loss(output, condition_class)
                loss.backward()
                e2optim.step()

                # 3.计算epsilon3项
                noise = np.random.normal(0, epsilon3)
                h.data = h.data + noise

            idx = idx + 1
            if idx >= samples:
                return


if __name__ == '__main__':
    sample()
    exit()
