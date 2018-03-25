import PIL.Image as Image
import torch
from torch.utils.data import *
import torchvision.transforms as transforms
import numpy as np
import os


# 定义数据集
class Placesdataset(Dataset):
    def __init__(self, path="place365_standard/", is_train=True, need_class_id=True, transforms=None):
        self.path = path  # 数据集位置
        self.need_class_id = need_class_id
        self.transforms = transforms

        if is_train:
            self.frame_file = path + "train.txt"
        else:
            self.frame_file = path + "val.txt"

        class2id = {}
        id2class = {}
        # 分类名称号码
        with open(self.path + "categories_places365.txt") as f:
            for eachline in f.readlines():
                c, cno = eachline.split()
                cname = "-".join(c.split('/')[2:])
                class2id[cname] = int(cno)
                id2class[int(cno)] = cname
        self.class2id = class2id
        self.id2class = id2class

        # 图片编号
        images = {}
        id = 0
        with open(self.frame_file) as f:
            for eachline in f.readlines():
                path = eachline.split()
                images[id] = path
                id = id + 1
        self.images = images

    def __len__(self):
        # 实现dataset长度
        return len(self.images)

    def __getitem__(self, idx):
        # 实现数据提取和转换
        path = self.images[idx]
        img = Image.open(self.path + path[0])
        img = img.convert('RGB')

        if self.transforms:
            img = self.transforms(img)

        class_id = self.class2id[path[0].split('/')[1]]
        sparse_class = torch.LongTensor([class_id])

        return {"img": img, "class": sparse_class}


def get_ImageNet_transform(resize_size=(224, 224), random_horizontal_flip=False, imagenet_normalize=True,
                           ordinary_normalize=False):
    trans = []
    trans.append(transforms.Resize(resize_size))
    if random_horizontal_flip:
        trans.append(transforms.RandomHorizontalFlip())

    trans.append(transforms.ToTensor())

    if imagenet_normalize and ordinary_normalize:
        raise Exception("imagenet_normalize and ordinary_normalize can't be True at the same time")

    # 用于分类识别的时候的归一化方式
    if imagenet_normalize:
        trans.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

    # ToTensor()会把张量归一化到[0,1],为了配合tanh,把数据归一化到[-1,1]
    if ordinary_normalize:
        trans.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    return transforms.Compose(trans)


# 存储模块参数
def save_param(module, path):
    torch.save(module.state_dict(), path)
    print("save params to {} successful!".format(path))


def restore_param(module, path):
    module.load_state_dict(torch.load(path))
    print("load params from {} successful!".format(path))


# 获得目前学习率
def get_learning_rate_from_scheduler_optim(scheduler_optiom):
    return scheduler_optiom.optimizer.param_groups[0]["lr"]


def get_learning_rate_from_optim(optim):
    return optim.param_groups[0]["lr"]


# 保存整个模型参数
def save_checkpoint(model_dict, global_step, now_batch, path):
    '''

    :param model_dict: 模型的参数，字典，key为某个保存的名字，value是模块
                        例：{
                            "generator_model":generator,
                            "discriminator_model":discriminator,
                            "optimizer":optimizer
                            }
                        最终会把generator的参数保存为"generator_model.pth"文件,以此类推

    :param global_step: 目前运行的梯度下降步数
    :param now_batch: 目前执行第几个batch
    :param path: 模型保存的地址
    :return: null
    '''
    if not path.endswith("/"):
        path = path + '/'

    for fname, module in model_dict.items():
        full_path = path + fname + ".pth"
        save_param(module, full_path)

    with open(path + "checkpoint", 'w') as f:
        f.writelines("global_step " + str(global_step) + '\n')
        f.writelines("now_batch " + str(now_batch) + '\n')


def restore_checkpoint(model_dict, path):
    '''

    :param model_dict: 模型的参数，字典，key为某个保存的名字，value是模块
                        例：{
                            "generator_model":generator,
                            "discriminator_model":discriminator,
                            "optimizer":optimizer
                            }
                        最终会加载"generator_model.pth"的参数导generator模块中，以此类推
    :param path: 模型保存的地址
    :return: global_step,now_batch
    '''

    if not path.endswith("/"):
        path = path + '/'

    for fname, module in model_dict.items():
        full_path = path + fname + ".pth"
        if not os.path.exists(full_path):
            print(full_path + " not found!")
        else:
            restore_param(module, full_path)

    if not os.path.exists(path + "checkpoint"):
        print("checkpoint not found")
        global_step = 0
        now_batch = 0
    else:
        with open(path + "checkpoint", "r")as f:
            fline = f.readlines()
            if len(fline) != 2:
                raise Exception("some params are missing")
            global_step = int(fline[0].split()[1])
            now_batch = int(fline[1].split()[1])

    return global_step, now_batch

