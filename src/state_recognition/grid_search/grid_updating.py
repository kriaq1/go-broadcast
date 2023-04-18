import numpy as np
import cv2


def get_graph_by_points(points, c=(1 + np.sqrt(2)) / 2):
    dists = np.sqrt(
        (-2 * np.dot(points, points.T)) + np.sum(points ** 2, axis=1, keepdims=True) + np.sum(points ** 2, axis=1))
    max_dist = np.max(dists)
    dists += np.eye(len(points)) * (max_dist + 1)
    graph = -np.ones((len(points), 4), dtype=int)
    min_dst = np.min(dists)
    lim_dst = c * min_dst
    for i in range(len(points)):
        for j, dst in enumerate(dists[i]):
            if min_dst <= dst <= lim_dst:
                if points[i][0] - points[j][0] > min_dst / 2:
                    graph[i][0] = j
                elif points[i][1] - points[j][1] > min_dst / 2:
                    graph[i][1] = j
                elif points[j][0] - points[i][0] > min_dst / 2:
                    graph[i][2] = j
                elif points[j][1] - points[i][1] > min_dst / 2:
                    graph[i][3] = j
    return graph


def get_graphs_components(graph):
    def dfs(graph, cur_comp, v, mask):
        mask[v] = cur_comp
        for n in graph[v]:
            if n != -1 and mask[n] == -1:
                dfs(graph, cur_comp, n, mask)

    components = -np.ones(len(graph), dtype=int)
    cur_comp = 0
    for i in range(len(graph)):
        if components[i] == -1:
            dfs(graph, cur_comp, i, components)
            cur_comp += 1
    return components


def update_positions(graph, v, positions, start=(0, 0)):
    def dfs_pos(graph, v, mask, value):
        mask[v] = 0
        positions[v] = value
        a = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]], dtype=int)
        for i, n in enumerate(graph[v]):
            if n != -1 and mask[n] == -1:
                nvalue = value + a[i]
                dfs_pos(graph, n, mask, nvalue)

    mask = -np.ones(len(graph), dtype=int)
    dfs_pos(graph, v, mask, np.array(start))


def is_triangle_integer_points(pt1, pt2, pt3):
    v, u = pt2 - pt1, pt3 - pt1
    return v[0] * u[1] != v[1] * u[0]


def get_indices_nearest_three_points(point, points, positions):
    sorted_pts = np.argsort(np.sum((points - point) ** 2, axis=1))
    i1 = 0
    i2 = 1
    for i3 in range(i2 + 1, len(sorted_pts)):
        if is_triangle_integer_points(positions[sorted_pts[i1]], positions[sorted_pts[i2]], positions[sorted_pts[i3]]):
            return sorted_pts[[i1, i2, i3]]


def get_nearest_position(point, points, positions):
    indices = get_indices_nearest_three_points(point, points, positions)
    assert indices is not None
    tri = points[indices].astype(np.float32)
    tri_pos = positions[indices].astype(np.float32)
    warp_mat = cv2.getAffineTransform(tri, tri_pos)
    pos = warp_mat @ np.hstack([point, 1])
    return np.rint(pos).astype(int)


def get_approximately_grid_point(point, points, positions):
    indices = get_indices_nearest_three_points(point, points, positions)
    tri = points[indices].astype(np.float32)
    tri_pos = positions[indices].astype(np.float32)
    warp_mat = cv2.getAffineTransform(tri, tri_pos)
    pos = warp_mat @ np.hstack([point, 1])
    pos = np.rint(pos)
    warp_mat = cv2.getAffineTransform(tri_pos, tri)
    return pos, warp_mat @ np.hstack([pos, 1])


def get_grid_by_step(shape, step=16):
    grid_shape = shape[0] // step - 1, shape[1] // step - 1
    grid = np.concatenate((np.arange(step, shape[0], step) * np.ones((1, grid_shape[0], 1)),
                           np.arange(step, shape[1], step).reshape((1, grid_shape[0], 1)) * np.ones(
                               grid_shape[1])), axis=0).transpose((2, 1, 0))
    grid = grid.reshape((grid_shape[0] * grid_shape[1], 2))
    return grid
