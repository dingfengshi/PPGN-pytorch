import torch.nn as nn
import torch



class modifiedAlexNet(nn.Module):

    def __init__(self, num_classes=365):
        super(modifiedAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        pool5 = self.features(x)
        x = pool5.view(x.size(0), 256 * 6 * 6)
        fc6 = self.classifier(x)
        return pool5.data, fc6.data


def get_weight(alexnet, path="whole_alexnet_places365_python36.pth.tar"):
    orgin = torch.load(path)
    o_state_dict = orgin.state_dict()
    pretrained_dicted = {k: v for k, v in o_state_dict.items() if k in alexnet.state_dict()}
    alexnet.load_state_dict(pretrained_dicted)
