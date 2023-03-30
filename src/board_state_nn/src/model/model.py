import torch
import torch.nn as nn
import timm


class Model(nn.Module):
    def __init__(self, **kwargs):
        self.out_classes = kwargs['out_classes']
        initialize_weights = kwargs['initialize_weights']
        super().__init__()
        # for name in timm.list_models():
            # print(name)
        self.mobnet = timm.create_model('mobilenetv3_small_050', pretrained=initialize_weights, num_classes=3*19*19)
        # self.resnet.global_pool = torch.nn.Identity()
        # self.resnet.classifier = torch.nn.Identity()
        # timm.models.MobileNetV3
        self.mobnet.forward_head = lambda x: x

        self.relu = nn.ReLU(True)
        self.conv = nn.Conv2d(288, 128, (1, 1), (1, 1), (0, 0), bias=True)
        self.fc = nn.Linear(128 * 19 * 19, 3 * 19 * 19)

    def forward(self, input):
        out = self.mobnet(input)

        out = self.conv(out)
        out = self.relu(out)

        out = torch.flatten(out, 1)
        out = self.fc(out)
        out = torch.unflatten(out, -1, (3, 19, 19))

        return out
