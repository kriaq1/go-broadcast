import torch
from ..find_board import segmentation_model
from .utils import approximation as approximation


class BoardSearch:
    def __init__(self, device=torch.device('cpu'), save_path='configs/model_saves/segmentation.pth'):
        self.device = device
        self.model = segmentation_model.load(save_name=save_path, device=self.device)

    def get_mask_image(self, image):
        assert image.shape == (1024, 1024, 3)
        return segmentation_model.get_mask_image(image=image, net=self.model, scale=0.5, device=self.device)

    def get_board_coordinates(self, image):
        mask = self.get_mask_image(image)
        return self.find_quadrilateral(mask)

    @staticmethod
    def find_quadrilateral(mask):
        assert mask.ndim == 2
        return approximation.approximate_polygon(mask)
