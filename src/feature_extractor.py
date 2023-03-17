import cv2
import numpy as np


def linear_perspective(image, points, shape=None):
    if shape is None:
        height = np.max(np.sqrt(np.sum((points - np.roll(points, 1)) ** 2, axis=0)))
        width = height
    else:
        height, width = shape
    output_points = np.float32([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]])
    input_points = np.float32(points)
    transformation = cv2.getPerspectiveTransform(input_points, output_points)
    return cv2.warpPerspective(image, transformation, (int(width), int(height)), flags=cv2.INTER_LINEAR)
