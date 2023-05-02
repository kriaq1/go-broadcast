import cv2
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt


def convert_cv_qt(cv_img):
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    p = convert_to_Qt_format.scaled(w, h, Qt.KeepAspectRatio)
    return QPixmap.fromImage(p)


def padding(image, shape=(1024, 1024), inter=cv2.INTER_AREA):
    old_height, old_width = image.shape[0], image.shape[1]
    if old_height / old_width <= shape[0] / shape[1]:
        new_width = shape[1]
        new_height = shape[1] * old_height // old_width
        image = cv2.resize(image, (new_width, new_height), interpolation=inter)
        y = shape[0] - image.shape[0]
        image = cv2.copyMakeBorder(image, y // 2, y // 2 + y % 2, 0, 0, cv2.BORDER_CONSTANT)
    else:
        new_width = shape[0] * old_width // old_height
        new_height = shape[0]
        image = cv2.resize(image, (new_width, new_height), interpolation=inter)
        x = shape[1] - image.shape[1]
        image = cv2.copyMakeBorder(image, 0, 0, x // 2, x // 2 + x % 2, cv2.BORDER_CONSTANT)
    return image


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
    return empty


def draw_board_state(board: np.ndarray, val=None, empty=None):
    if empty is None:
        empty = draw_empty_board()
        coords = np.linspace(28, 580, 19).astype(int)
    for i in range(len(coords)):
        for j in range(len(coords)):
            x = coords[i]
            y = coords[j]
            if val:
                xshift = 5 * len(str(val[j][i]))
            if board[j][i] == 1:
                cv2.circle(empty, (x, y + 2), radius=12, color=(50, 50, 50), thickness=-1)
                cv2.circle(empty, (x, y), radius=12, color=(255, 255, 255), thickness=-1)
                if val:
                    cv2.putText(empty, str(val[j][i]), (x - xshift, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 0), 1, cv2.LINE_AA)
            if board[j][i] == -1:
                cv2.circle(empty, (x, y + 2), radius=12, color=(100, 100, 100), thickness=-1)
                cv2.circle(empty, (x, y), radius=12, color=(0, 0, 0), thickness=-1)
                if val:
                    cv2.putText(empty, str(val[j][i]), (x - xshift, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1, cv2.LINE_AA)
    return empty

