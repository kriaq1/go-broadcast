import cv2
import numpy as np
import math


def approximate_polygon(mask, hull=True):
    if hull:
        return approximate_polygon_hull(mask)
    return approximate_polygon_poly_dp(mask)


def poly_dp_search(contour):
    coefficient = 0.0003
    approx = None
    while approx is None or len(approx) > 4:
        approx = cv2.approxPolyDP(contour, coefficient * cv2.arcLength(contour, True), True)
        coefficient += 0.0003
    return approx


def approximate_polygon_poly_dp(mask):
    assert mask.ndim == 2
    mask = np.array(mask, dtype=np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    selected_contour = max(contours, key=lambda x: cv2.contourArea(x))
    return poly_dp_search(selected_contour)


def get_angle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return -1, -1

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def get_triangle_area(a, b, c):
    return 0.5 * abs(((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])))


def approximate_polygon_hull(mask, n: int = 4) -> list[(int, int)]:
    assert mask.ndim == 2
    mask = np.array(mask, dtype=np.uint8)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    max_contour = max(contours, key=lambda x: cv2.contourArea(x))
    hull = cv2.convexHull(max_contour)
    hull = list(np.array(hull).reshape((len(hull), 2)))
    while len(hull) > n:
        best = [1e8, 0, 1, (0, 0)]
        for edge_index_1 in range(len(hull)):
            edge_index_2 = (edge_index_1 + 1) % len(hull)

            close_edge_index_1 = (edge_index_1 - 1) % len(hull)
            close_edge_index_2 = (edge_index_1 + 2) % len(hull)

            edge_point_1 = hull[edge_index_1]
            edge_point_2 = hull[edge_index_2]
            close_edge_point_1 = hull[close_edge_index_1]
            close_edge_point_2 = hull[close_edge_index_2]

            angle1 = get_angle(close_edge_point_1, edge_point_1, edge_point_2)
            angle2 = get_angle(edge_point_1, edge_point_2, close_edge_point_2)

            if angle1 + angle2 <= math.pi:
                continue
            intersect = line_intersection((close_edge_point_1, edge_point_1), (edge_point_2, close_edge_point_2))
            if intersect[0] < 0 or intersect[0] > 1024 or intersect[1] < 0 or intersect[1] > 1024:
                continue
            area = get_triangle_area(edge_point_1, intersect, edge_point_2)
            if best is None or best[0] > area:
                best = (area, edge_index_1, edge_index_2, intersect)
        _, edge_index_1, edge_index_2, intersect = best
        hull[edge_index_1] = intersect
        del hull[edge_index_2]
    hull = [(int(x), int(y)) for x, y in hull]
    return hull
