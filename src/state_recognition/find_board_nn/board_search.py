import torch
import numpy as np
from . import segmentation_model
from .utils import approximation


class BoardSearch:
    def __init__(self, device=torch.device('cpu'), save_path='configs/model_saves/segmentation.pth'):
        self.device = device
        self.model = segmentation_model.load(save_name=save_path, device=self.device)

    def get_mask_image(self, image, out_threshold=0.5):
        return segmentation_model.get_mask_image(image=image, net=self.model, scale=0.5, device=self.device,
                                                 out_threshold=out_threshold)

    def get_board_coordinates(self, image, out_threshold=0.5):
        mask = self.get_mask_image(image, out_threshold=out_threshold)
        return self.find_quadrilateral(mask)

    @staticmethod
    def find_quadrilateral(mask):
        coordinates = np.array(approximation.approximate_polygon(mask))[::-1]
        return np.roll(coordinates, -np.argmin(np.sum(coordinates ** 2, axis=1)), axis=0)
