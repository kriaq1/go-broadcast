import numpy as np
import torch
import cv2


def open_input(filename):
    path = str(filename)
    image = cv2.imread(path)
    assert image.ndim == 3
    return image


def open_target(filename):
    path = str(filename)
    board = np.load(path)
    assert board.dtype == 'int64'
    assert board.shape == (19, 19)
    return board


def preprocess_input(input):
    size = 608
    input = cv2.resize(input, (size, size), interpolation=cv2.INTER_CUBIC)
    assert input.shape == (size, size, 3)
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)

    input = input / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = (input - mean) / std

    input = input.transpose((2, 0, 1))
    return input


def preprocess_target(target):
    board = target
    board += 1
    assert np.max(board) <= 2 and np.min(board) >= 0
    board = torch.as_tensor(board)
    return torch.nn.functional.one_hot(board, 3).permute(2, 0, 1)
