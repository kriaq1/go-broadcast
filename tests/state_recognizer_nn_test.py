import matplotlib.pyplot as plt
from os import listdir

import cv2
import torch
import numpy as np

from src.feature_extractor import linear_perspective

from src.find_board_nn import BoardSearch
from src.find_board_nn import load_image

from src.board_state_nn import BoardStateRecognizer


def show_board_state(image, board):
    cv2.imshow('image', image)
    board = board[::-1]
    empty = np.zeros((608, 608, 3), np.uint8)
    empty[:, :] = (0, 100, 255)
    coords = np.linspace(28, 580, 19).astype(int)
    for i in range(len(coords)):
        for j in range(len(coords)):
            x = coords[i]
            y = coords[j]
            if board[j][i] == 1:
                empty = cv2.circle(empty, (x, y), radius=10, color=(255, 255, 255), thickness=-1)
            if board[j][i] == -1:
                empty = cv2.circle(empty, (x, y), radius=10, color=(0, 0, 0), thickness=-1)
    cv2.imshow('board', empty)
    cv2.waitKey()


if __name__ == '__main__':
    save_path_search = '../configs/model_saves/segmentation.pth'
    save_path_recognizer = '../configs/model_saves/board_state.pth'
    test_path = 'predict_images/original/'
    result_cut_path = 'predict_images/cut/'
    result_predict_path = 'predict_images/predict/'

    device = torch.device('cpu')
    # if you can use cuda:
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    search = BoardSearch(device=device, save_path=save_path_search)
    state_recognizer = BoardStateRecognizer(device=device, save_path=save_path_recognizer)

    for file in listdir(test_path):
        image = load_image(test_path + str(file))
        points = search.get_board_coordinates(image)
        cut = linear_perspective(image, points, 608)
        board = state_recognizer.get_predict(cut)
        cv2.imwrite(result_cut_path + file, cut)
        np.savetxt(result_predict_path + file[:-4] + '.txt', board[::-1], fmt='%d')
        show_board_state(cut, board)
