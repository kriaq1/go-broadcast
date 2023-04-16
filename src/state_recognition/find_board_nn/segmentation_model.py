import torch

from .model import Model
from .utils.predict import predict_image
from .utils.utils import mask_to_image


def load(save_name='model.pth', device=torch.device('cpu')):
    net = Model(in_channels=3, out_classes=1)
    net.to(device=device)
    state_dict = torch.load(save_name, map_location=device)
    del state_dict['mask_values']
    net.load_state_dict(state_dict)
    return net


def get_mask_image(image, net, scale=0.5, device=torch.device('cpu')):
    mask = predict_image(image, net, device, scale)
    return mask_to_image(mask, [0, 255]).astype(int)


def load_and_predict(image, save_name='model.pth', scale=0.5, device=torch.device('cpu')):
    net = load(save_name)
    return get_mask_image(image, net, scale, device)
