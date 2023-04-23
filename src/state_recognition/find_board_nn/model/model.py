import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class Model(nn.Module):
    def __init__(self, in_channels, out_classes, **kwargs):
        self.in_channels = in_channels
        self.out_classes = out_classes
        arch = 'FPN'
        encoder_name = 'resnet18'

        super().__init__()
        self.model = smp.create_model(
            arch=arch, encoder_name=encoder_name, encoder_weights=None,
            in_channels=in_channels, classes=out_classes, **kwargs
        )

        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer('std', torch.tensor(params['std']).view(1, 3, 1, 1))
        self.register_buffer('mean', torch.tensor(params['mean']).view(1, 3, 1, 1))

    def forward(self, image):
        image = (image - self.mean) / self.std
        return self.model(image)
