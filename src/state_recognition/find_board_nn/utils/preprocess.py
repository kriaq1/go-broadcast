import cv2

from .preprocess_utils import padding


def load_image(path):
    image = cv2.imread(path)
    if image.shape[:2] != (1024, 1024):
        image = padding(image, 1024, inter=cv2.INTER_CUBIC)
    assert image.shape == (1024, 1024, 3)
    return image


def preprocess(image, scale=0.5):
    weight, height = image.shape[1], image.shape[0]
    new_weight, new_height = int(scale * weight), int(scale * height)
    assert new_weight > 0 and new_height > 0
    image = cv2.resize(image, (new_weight, new_height), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))
    if (image > 1).any():
        image = image / 255.0
    return image
