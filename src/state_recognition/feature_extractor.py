import cv2
import numpy as np


def linear_perspective(image, points, shape=1024):
    height, width = shape, shape
    output_points = np.float32([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]])
    input_points = np.float32(points)
    transformation = cv2.getPerspectiveTransform(input_points, output_points)
    return cv2.warpPerspective(image, transformation, (int(width), int(height)), flags=cv2.INTER_LINEAR)
