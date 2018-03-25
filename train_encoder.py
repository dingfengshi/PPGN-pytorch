import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as datas
from tensorboardX import SummaryWriter
import torchvision
from torchvision.models.alexnet import *

import DataUtill

# 训练参数
learning_rate = 0.01
epochs = 6000
batch_size = 256
model_path = "model/"
new_model_path = "model2/"
data_path = "/dev/shm/place365_standard"


def eval(trans, encoder, critizen):
    # 返回acc%
    valid_data = DataUtill.Placesdataset(data_path, transforms=trans, is_train=False)
    valid_data_loader = datas.DataLoader(valid_data, batch_size, shuffle=False, num_workers=1)

    match_count = 0
    data_num = 0
    total_loss = 0
    batch_count = 0
    encoder.eval()
    for step, valid_batch in enumerate(valid_data_loader):
        batch_count = batch_count + 1
        valid_img = autograd.Variable(valid_batch["img"].cuda())
        valid_label = autograd.Variable(valid_batch["class"].cuda())

        output = encoder(valid_img)
        _, idx = torch.max(output, 1)
        tar = valid_label.squeeze()
        total_loss += critizen(output, tar).data[0]
        match_count += ((idx == tar).sum().float()).data[0]
        data_num += tar.size(0)

    encoder.train()
    return (match_count / data_num) * 100, total_loss / batch_count


def train(model_path, epochs):
    trans = DataUtill.get_ImageNet_transform(random_horizontal_flip=True)
    train_data = DataUtill.Placesdataset(data_path, transforms=trans)
    train_data_loader = datas.DataLoader(train_data, batch_size, shuffle=True, num_workers=8)

    # 可视化数据
    # torchvision.utils.save_image(valid_batch["img"], "pic.png", normalize=True)

    encoder = alexnet(True)
    num_fea = encoder.classifier[6].in_features
    features = list(encoder.classifier.children())[:-1]
    ofc = nn.Linear(num_fea, 200)
    nn.init.normal(ofc.weight, 0, 0.01)
    features.append(ofc)
    encoder.classifier = nn.Sequential(*features)

    encoder = encoder.cuda()
    global_step = 0

    optimizer = optim.SGD(encoder.parameters(), learning_rate, 0.9, weight_decay=0.0005)
    optimizer = optim.lr_scheduler.ExponentialLR(optimizer, 0.998)
    critizen = nn.CrossEntropyLoss()

    Writer = SummaryWriter(log_dir=model_path)

    # 先计算一次当前acc
    max_acc, min_eval_loss = eval(trans, encoder, critizen)
    print("初始准确率为{}%".format(max_acc))
    Writer.add_scalar("/eval/eval_loss", min_eval_loss, global_step)
    Writer.add_scalar("/eval/accuracy", max_acc, global_step)

    for epoch in range(epochs):
        for step, batch in enumerate(train_data_loader):
            global_step = global_step + 1
            input = batch["img"]
            label = batch["class"]

            # torchvision.utils.save_image(input, "pic.png", normalize=True)

            input = autograd.Variable(input)
            label = autograd.Variable(label)

            input = input.cuda()
            label = label.cuda()

            encoder.zero_grad()
            output = encoder(input)
            loss = critizen(output, label.squeeze())
            loss.backward(retain_graph=True)
            optimizer.step()

            if global_step % 100 == 0:
                Writer.add_scalar("train_loss", loss, global_step)

            if global_step % 1000 == 0:
                Writer.add_histogram("/conv1/grad", encoder.features[0].weight.grad, global_step)
                Writer.add_histogram("/conv1/weight", encoder.features[0].weight, global_step)
                Writer.add_histogram("/fc6/grad", encoder.classifier[6].weight.grad, global_step)
                Writer.add_histogram("/fc6/weight", encoder.classifier[6].weight, global_step)
                acc, eval_loss = eval(trans, encoder, critizen)
                Writer.add_scalar("/eval/accuracy", acc, global_step)
                Writer.add_scalar("/eval/eval_loss", eval_loss, global_step)
                if acc > max_acc:
                    max_acc = max(acc, max_acc)
                    DataUtill.save_param(encoder, model_path + "alexnet.pkl")
                    print(
                        "save params in {} epoch {} step with accuracy {}% , and the loss is {}".format(epoch, step,
                                                                                                        acc, eval_loss))


def get_trained_encoder(model_path="model/alexnet.pkl"):
    encoder = alexnet(False)
    num_fea = encoder.classifier[6].in_features
    features = list(encoder.classifier.children())[:-1]
    ofc = nn.Linear(num_fea, 200)
    nn.init.xavier_normal(ofc.weight)
    features.append(ofc)
    encoder.classifier = nn.Sequential(*features)
    DataUtill.restore_param(encoder, model_path)
    return encoder


if __name__ == '__main__':
    train(model_path=model_path, epochs=epochs)
