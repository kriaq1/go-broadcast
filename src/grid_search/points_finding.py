import cv2
import numpy as np
import torch


def get_points_by_matching(match, threshold=0, u=(21, 21)):
    match = torch.as_tensor(match)
    match = match.unsqueeze(0)
    match = match.unsqueeze(0)
    pooled_match = torch.nn.functional.max_pool2d(match, u, stride=1, padding=(u[0] // 2, u[1] // 2))
    match = match.squeeze().numpy()
    pooled_match = pooled_match.squeeze().numpy()
    return np.array(list(zip(*np.where(np.logical_and(match == pooled_match, match >= threshold)))))


def neighborhood(img, point, u=(21, 21)):
    u = np.array(u)
    start = np.maximum(point - u // 2, [0, 0])
    end = np.minimum(point + u // 2 + 1, np.array(img.shape[0:2]))
    return img[start[0]:end[0], start[1]:end[1]]


def get_checked_corners(gray, pts, u=(21, 21), e=(5, 5), threshold=5 * 10 ** (-6), blockSize=2, ksize=3, k=0.04):
    corners = []
    for pt in pts:
        nb = neighborhood(gray, pt, u)
        dst = cv2.cornerHarris(nb, blockSize, ksize, k)
        # print(pt, np.abs(np.array(u) // 2 - np.unravel_index(np.argmax(dst), u)), np.max(dst), np.min(dst))
        if np.max(dst) >= threshold and np.all(
                np.abs(np.array(u) // 2 - np.unravel_index(np.argmax(dst), u)) <= np.array(e)):
            corners.append(pt)
    return np.array(corners)


