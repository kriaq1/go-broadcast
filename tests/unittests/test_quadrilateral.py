import pytest
import numpy as np
import cv2
from src.state_recognition.find_board_nn.utils.approximation import approximate_polygon


def assert_close(points1, points2, epsilon=10):
    points1 = np.roll(points1, -np.argmin(np.sum(points1 ** 2, axis=1)), axis=0)
    points2 = np.roll(points2, -np.argmin(np.sum(points2 ** 2, axis=1)), axis=0)

    assert np.sum(np.abs(points1 - points2)) < epsilon


@pytest.mark.test_quadrilateral
def test_hull_approximation():
    mask = np.zeros((100, 100))
    points = np.array([[30, 30], [40, 60], [60, 60], [60, 25]])
    cv2.fillPoly(mask, pts=[points], color=255)

    approx_points = np.array(approximate_polygon(mask, hull=True))[::-1]
    assert_close(points, approx_points)


@pytest.mark.test_empty
def test_empty_approximation():
    mask = np.zeros((50, 50))
    with pytest.raises(ValueError):
        approximate_polygon(mask, hull=True)


@pytest.mark.test_quadrilateral
def test_poly_dp_approximation():
    mask = np.zeros((121, 141))
    points = np.array([[34, 43], [43, 73], [65, 65], [66, 22]])
    cv2.fillPoly(mask, pts=[points], color=255)

    approx_points = np.array(approximate_polygon(mask, hull=False)).reshape(4, 2)
    assert_close(points, approx_points)
