import cv2
import numpy as np
import sympy


def approximate_polygon(mask):
    return approximate_polygon_poly_db(mask)


def poly_dp_binsearch(contour):
    coefficient = 0.0001
    approx = None
    while approx is None or len(approx) > 4:
        approx = cv2.approxPolyDP(contour, coefficient * cv2.arcLength(contour, True), True)
        coefficient += 0.0001
    return approx


def approximate_polygon_poly_db(mask):
    assert mask.ndim == 2
    mask = np.array(mask, dtype=np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    selected_contour = max(contours, key=lambda x: cv2.contourArea(x))
    return poly_dp_binsearch(selected_contour)


def approximate_polygon_hull(mask, n: int = 4) -> list[(int, int)]:
    assert mask.ndim == 2
    mask = np.array(mask, dtype=np.uint8)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    hull = cv2.convexHull(contours[0])
    hull = np.array(hull).reshape((len(hull), 2))
    hull = [sympy.Point(*pt) for pt in hull]
    while len(hull) > n:
        best_candidate = None
        for edge_idx_1 in range(len(hull)):
            edge_idx_2 = (edge_idx_1 + 1) % len(hull)

            adj_idx_1 = (edge_idx_1 - 1) % len(hull)
            adj_idx_2 = (edge_idx_1 + 2) % len(hull)

            edge_pt_1 = sympy.Point(*hull[edge_idx_1])
            edge_pt_2 = sympy.Point(*hull[edge_idx_2])
            adj_pt_1 = sympy.Point(*hull[adj_idx_1])
            adj_pt_2 = sympy.Point(*hull[adj_idx_2])

            subpoly = sympy.Polygon(adj_pt_1, edge_pt_1, edge_pt_2, adj_pt_2)
            angle1 = subpoly.angles[edge_pt_1]
            angle2 = subpoly.angles[edge_pt_2]
            if sympy.N(angle1 + angle2) <= sympy.pi:
                continue
            adj_edge_1 = sympy.Line(adj_pt_1, edge_pt_1)
            adj_edge_2 = sympy.Line(edge_pt_2, adj_pt_2)
            intersect = adj_edge_1.intersection(adj_edge_2)[0]
            area = sympy.N(sympy.Triangle(edge_pt_1, intersect, edge_pt_2).area)
            if best_candidate and best_candidate[1] < area:
                continue
            better_hull = list(hull)
            better_hull[edge_idx_1] = intersect
            del better_hull[edge_idx_2]
            best_candidate = (better_hull, area)

        if not best_candidate:
            raise ValueError("Could not find the best fit n-gon!")

        hull = best_candidate[0]
    hull = [(int(x), int(y)) for x, y in hull]
    return hull
