import numpy as np


def mask_to_image(mask: np.ndarray):
    out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)
    out[mask == 1] = 255
    return out
