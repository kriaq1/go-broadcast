import matplotlib.pyplot as plt
from os import listdir

import cv2
import torch
import numpy as np

from src.find_board_nn import BoardSearch
from src.find_board_nn import load_image


def plot_image_and_mask(image, mask, points, save_name=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    classes = 2
    fig, ax = plt.subplots(1, classes + 1)

    ax[0].set_title('Original')
    ax[0].imshow(image)

    filled_image = image.copy()
    filled_image[mask == 255] = (0, 255, 0)
    ax[1].set_title('Neural network')
    ax[1].imshow(filled_image)

    contour_image = cv2.drawContours(image, [np.array(points)], -1, (255, 0, 0), 10)
    ax[2].set_title('Find quadrilateral')
    ax[2].imshow(contour_image)

    plt.xticks([]), plt.yticks([])
    if save_name is None:
        plt.show()
    else:
        plt.savefig('find_board_data/result/' + save_name, format='jpg', dpi=1024)
    plt.close(fig)


if __name__ == '__main__':
    save_path = '../configs/model_saves/segmentation.pth'
    test_path = 'find_board_data/test/'
    result_path = 'find_board_data/result/'

    device = torch.device('cpu')
    # if you can use cuda:
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    search = BoardSearch(device=device, save_path=save_path)

    for file in listdir(test_path):
        image = load_image(test_path + str(file))
        mask = search.get_mask_image(image)
        points = search.find_quadrilateral(mask)
        plot_image_and_mask(image, mask, points, str(file))
