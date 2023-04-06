import cv2
import numpy as np
import torch
from . import points_finding
from . import grid_updating


class GridSearch:
    def __init__(self, template=None, path=None):
        if template is None:
            template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            res, template = cv2.threshold(template, 0, 255, cv2.THRESH_OTSU)
        self.grid = None
        self.image_shape = None
        self.points = None
        self.positions = None
        if len(template.shape) > 2:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        res, self.template = cv2.threshold(template, 0, 255, cv2.THRESH_OTSU)
        self.tshape = template.shape
        self.template_center = (template.shape[0] // 2, template.shape[1] // 2)
        self.check_corners = True
        self.distance_coefficient = (1 + np.sqrt(2)) / 2
        self.matching_thresh = 0
        self.kernel_size = (2 * self.tshape[0] + 1, 2 * self.tshape[1] + 1)
        self.corner_checking_size = self.tshape
        self.nearest_corner = (5, 5)
        self.harris_threshold = 5e-6
        self.harris_block_size = 2
        self.harris_ksize = 3
        self.harris_k = 0.04
        self.grid_step = 16
        self.error_number = np.iinfo(np.int64).max // 2

    def get_found_points(self, image):
        self.image_shape = image.shape[0:2]
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        matching = cv2.matchTemplate(image, self.template, cv2.TM_CCOEFF_NORMED)
        points = points_finding.get_points_by_matching(matching, self.matching_thresh,
                                                       self.kernel_size)
        points += self.template_center
        if self.check_corners:
            points = points_finding.get_checked_corners(image, points, self.corner_checking_size, self.nearest_corner,
                                                        self.harris_threshold, self.harris_block_size,
                                                        self.harris_ksize, self.harris_k)
        self.points = points
        return points

    def get_points_positions(self, image=None):
        if image is not None:
            if len(image.shape) > 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.get_found_points(image)
        graph = grid_updating.get_graph_by_points(self.points, self.distance_coefficient)
        components = grid_updating.get_graphs_components(graph)
        biggest_comp_n = np.argmax(np.bincount(components))
        self.positions = np.ones(self.points.shape, dtype=int) * np.iinfo(np.int64).max // 2
        grid_updating.update_positions(graph, np.where(components == biggest_comp_n)[0][0], self.positions, (0, 0))
        positions_set = set(zip(self.positions[:, 0], self.positions[:, 1]))
        for i in range(len(self.positions)):
            if components[i] == biggest_comp_n:
                continue
            position = grid_updating.get_nearest_position(self.points[i], self.points[components == biggest_comp_n],
                                                          self.positions[components == biggest_comp_n])
            assert tuple(position) not in positions_set
            grid_updating.update_positions(graph, i, self.positions, position)
            components[components == components[i]] = biggest_comp_n
        return self.positions, self.points

    def build_grid(self, image=None):
        if image is not None:
            if len(image.shape) > 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                self.get_points_positions(image)
        new_points = []
        new_positions = []

        positions_set = set(zip(self.positions[:, 0], self.positions[:, 1]))
        checking_points = grid_updating.get_grid_by_step(self.image_shape, self.grid_step)

        for point in checking_points:
            position, appr_point = grid_updating.get_approximately_grid_point(point, self.points, self.positions)
            if tuple(position) not in positions_set:
                new_positions.append(position)
                new_points.append(appr_point)
                positions_set.add(tuple(position))
        self.positions = np.vstack([self.positions, new_positions]).astype(int)
        self.points = np.vstack([self.points, new_points]).astype(int)
        mask = self.get_removing_mask()
        self.points = self.points[mask]
        self.positions = self.positions[mask]
        self.positions -= np.min(self.positions[:, 0]), np.min(self.positions[:, 1])
        self.grid = np.ones((np.max(self.positions[:, 0]) + 1, np.max(self.positions[:, 1]) + 1, 2),
                            dtype=int) * np.iinfo(int).max // 2
        for point, position in zip(self.points, self.positions):
            self.grid[position[0]][position[1]] = point
        self.cut_grid((19, 19))
        return self.grid

    def get_removing_mask(self):
        out_of_bound = np.logical_and(np.all(0 <= self.points, axis=1), np.all(self.points < self.image_shape, axis=1))
        return out_of_bound

    def cut_grid(self, shape=(19, 19)):
        vmask, hmask = np.ones(self.grid.shape[0], dtype=bool), np.ones(self.grid.shape[1], dtype=bool)
        vmask[np.sum(np.all(self.grid != self.error_number, axis=2), axis=1) < shape[1]] = False
        hmask[np.sum(np.all(self.grid != self.error_number, axis=2), axis=0) < shape[0]] = False
        self.grid = self.grid[vmask.reshape((vmask.shape[0], 1)) * hmask].reshape(
            (np.sum(vmask), np.sum(hmask), self.grid.shape[2]))

    def set_parameters(self, **kwargs):
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
        pass
