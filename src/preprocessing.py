import cv2
import numpy as np


def thresholding(image, threshold=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if threshold is None:
        return cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
    else:
        return cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]


def adaptive_thresholding(image, a=31, b=5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, a, b)


def padding(image, size=512):
    old_height, old_width = image.shape[0], image.shape[1]
    if old_height <= old_width:
        new_width = size
        new_height = size * old_height // old_width
        image = cv2.resize(image, (new_width, new_height))
        y = size - image.shape[0]
        image = cv2.copyMakeBorder(image, y // 2, y // 2 + y % 2, 0, 0, cv2.BORDER_CONSTANT)
    else:
        new_width = size * old_width // old_height
        new_height = size
        image = cv2.resize(image, (new_width, new_height))
        x = size - image.shape[1]
        image = cv2.copyMakeBorder(image, 0, 0, x // 2, x // 2 + x % 2, cv2.BORDER_CONSTANT)
    return image


def remove_shadows(image):
    rgb_planes = cv2.split(image)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
    result_norm = cv2.merge(result_norm_planes)
    return result_norm
