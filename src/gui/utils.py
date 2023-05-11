import cv2
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import yaml
from yaml.loader import SafeLoader
import os


def dict_by_parameters(parameters):
    if type(parameters) == str:
        if os.path.isfile(parameters):
            with open(parameters, 'r') as fd:
                parameters = yaml.load(fd, Loader=SafeLoader)
    return parameters


def load_dict(dct, parameters):
    is_yaml = (type(parameters) == str) and os.path.isfile(parameters)
    parameters_dict = parameters
    if is_yaml:
        with open(parameters, 'r') as fd:
            parameters_dict = yaml.load(fd, SafeLoader)
    for key in parameters_dict.keys():
        if dct[key] is not None:
            parameters_dict[key] = dct[key]
    if is_yaml:
        with open(parameters, 'w') as fd:
            yaml.dump(parameters_dict, fd)


def convert_cv_qt(cv_img):
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    p = convert_to_Qt_format.scaled(w, h, Qt.KeepAspectRatio)
    return QPixmap.fromImage(p)


def padding(image, shape=(1024, 1024), color=(0, 0, 0), inter=cv2.INTER_AREA):
    old_height, old_width = image.shape[0], image.shape[1]
    if old_height * shape[1] <= shape[0] * old_width:
        new_width = shape[1]
        new_height = shape[1] * old_height // old_width
        image = cv2.resize(image, (new_width, new_height), interpolation=inter)
        y = shape[0] - image.shape[0]
        image = cv2.copyMakeBorder(image, y // 2, y // 2 + y % 2, 0, 0, cv2.BORDER_CONSTANT, value=color)
    else:
        new_width = shape[0] * old_width // old_height
        new_height = shape[0]
        image = cv2.resize(image, (new_width, new_height), interpolation=inter)
        x = shape[1] - image.shape[1]
        image = cv2.copyMakeBorder(image, 0, 0, x // 2, x // 2 + x % 2, cv2.BORDER_CONSTANT, value=color)
    return image


def unpadding_points(points, shape, padded_shape=(1024, 1024)):
    points = points.astype(np.float64)
    old_height, old_width = shape[0], shape[1]
    if shape[0] * padded_shape[1] <= padded_shape[0] * shape[1]:
        new_width, new_height = padded_shape[1], padded_shape[1] * old_height // old_width
        points -= [0, (padded_shape[0] - new_height) // 2]
        points *= old_width / new_width
    else:
        new_width, new_height = padded_shape[0] * old_width // old_height, padded_shape[0]
        points -= [(padded_shape[1] - new_width) // 2, 0]
        points *= old_height / new_height
    return points.astype(int)


def padding_points(points, shape, padded_shape=(1024, 1024)):
    points = points.astype(np.float64)
    old_height, old_width = shape[0], shape[1]
    if shape[0] * padded_shape[1] <= padded_shape[0] * shape[1]:
        new_width, new_height = padded_shape[1], padded_shape[1] * old_height // old_width
        points *= new_height / old_height
        points += [0, (padded_shape[0] - new_height) // 2]

    else:
        new_width, new_height = padded_shape[0] * old_width // old_height, padded_shape[0]
        points *= new_width / old_width
        points += [(padded_shape[1] - new_width) // 2, 0]
    return points.astype(int)


def draw_contours(image: np.ndarray, points, color=(0, 0, 255), thickness=2) -> np.ndarray:
    return cv2.drawContours(image, [np.array(points)], -1, color, thickness)


def draw_empty_board():
    empty = np.zeros((608, 608, 3), np.uint8)
    empty[:, :] = (181, 217, 253)
    coords = np.linspace(28, 580, 19).astype(int)
    for i in range(len(coords)):
        cv2.line(empty, (coords[0], coords[i]), (coords[-1], coords[i]), (0, 0, 0), thickness=1)
        cv2.line(empty, (coords[i], coords[0]), (coords[i], coords[-1]), (0, 0, 0), thickness=1)
    for i in range(3):
        for j in range(3):
            x = coords[3 + 6 * i]
            y = coords[3 + 6 * j]
            cv2.circle(empty, (x, y), radius=3, color=(0, 0, 0), thickness=-1)
    for i, coord in enumerate(coords):
        char = chr(ord('A') + i + int(i > 7))
        cv2.putText(empty, char, (coord - 5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(empty, str(19 - i), (8 - int(i < 10) * 4, coord + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return empty


def draw_board_state(board: np.ndarray, val=None, empty=None):
    if empty is None:
        empty = draw_empty_board()
    coords = np.linspace(28, 580, 19).astype(int)
    for i in range(len(coords)):
        for j in range(len(coords)):
            state = int(board[j][i])
            if state == 0:
                continue
            x = int(coords[i])
            y = int(coords[j])
            if state == 1:
                color = (255, 255, 255)
                shadow_color = (50, 50, 50)
                text_color = (0, 0, 0)
            elif state == -1:
                color = (0, 0, 0)
                shadow_color = (100, 100, 100)
                text_color = (255, 255, 255)
            else:
                color = (0, 0, 255)
                shadow_color = (0, 0, 255)
                text_color = (0, 0, 0)
            cv2.circle(empty, (x, y + 2), radius=12, color=shadow_color, thickness=-1)
            cv2.circle(empty, (x, y), radius=12, color=color, thickness=-1)
            if val is None:
                continue
            value = val[j][i]
            if value > 0:
                str_value = str(value)
                xshift = 4 * len(str_value)
                cv2.putText(empty, str(str_value), (x - xshift, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            text_color, 1, cv2.LINE_AA)
            else:
                cv2.circle(empty, (x, y + 2), radius=12, color=(0, 0, 255), thickness=2)

    return empty
