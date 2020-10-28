#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: endtoend_env_utils.py
# =====================================

import math

import numpy as np


def shift_coordination(orig_x, orig_y, coordi_shift_x, coordi_shift_y):
    '''
    :param orig_x: original x
    :param orig_y: original y
    :param coordi_shift_x: coordi_shift_x along x axis
    :param coordi_shift_y: coordi_shift_y along y axis
    :return: shifted_x, shifted_y
    '''
    shifted_x = orig_x - coordi_shift_x
    shifted_y = orig_y - coordi_shift_y
    return shifted_x, shifted_y


def rotate_coordination(orig_x, orig_y, orig_d, coordi_rotate_d):
    """
    :param orig_x: original x
    :param orig_y: original y
    :param orig_d: original degree
    :param coordi_rotate_d: coordination rotation d, positive if anti-clockwise, unit: deg
    :return:
    transformed_x, transformed_y, transformed_d(range:(-180 deg, 180 deg])
    """

    coordi_rotate_d_in_rad = coordi_rotate_d * math.pi / 180
    transformed_x = orig_x * math.cos(coordi_rotate_d_in_rad) + orig_y * math.sin(coordi_rotate_d_in_rad)
    transformed_y = -orig_x * math.sin(coordi_rotate_d_in_rad) + orig_y * math.cos(coordi_rotate_d_in_rad)
    transformed_d = orig_d - coordi_rotate_d
    if transformed_d > 180:
        while transformed_d > 180:
            transformed_d = transformed_d - 360
    elif transformed_d <= -180:
        while transformed_d <= -180:
            transformed_d = transformed_d + 360
    else:
        transformed_d = transformed_d
    return transformed_x, transformed_y, transformed_d


def shift_and_rotate_coordination(orig_x, orig_y, orig_d, coordi_shift_x, coordi_shift_y, coordi_rotate_d):
    shift_x, shift_y = shift_coordination(orig_x, orig_y, coordi_shift_x, coordi_shift_y)
    transformed_x, transformed_y, transformed_d \
        = rotate_coordination(shift_x, shift_y, orig_d, coordi_rotate_d)
    return transformed_x, transformed_y, transformed_d


def rotate_and_shift_coordination(orig_x, orig_y, orig_d, coordi_shift_x, coordi_shift_y, coordi_rotate_d):
    shift_x, shift_y, transformed_d \
        = rotate_coordination(orig_x, orig_y, orig_d, coordi_rotate_d)
    transformed_x, transformed_y = shift_coordination(shift_x, shift_y, coordi_shift_x, coordi_shift_y)

    return transformed_x, transformed_y, transformed_d


def cal_info_in_transform_coordination(filtered_objects, x, y, rotate_d):  # rotate_d is positive if anti
    results = []
    for obj in filtered_objects:
        orig_x = obj['x']
        orig_y = obj['y']
        orig_v = obj['v']
        orig_heading = obj['phi']
        width = obj['w']
        length = obj['l']
        route = obj['route']
        shifted_x, shifted_y = shift_coordination(orig_x, orig_y, x, y)
        trans_x, trans_y, trans_heading = rotate_coordination(shifted_x, shifted_y, orig_heading, rotate_d)
        trans_v = orig_v
        results.append({'x': trans_x,
                        'y': trans_y,
                        'v': trans_v,
                        'phi': trans_heading,
                        'w': width,
                        'l': length,
                        'route': route,})
    return results


def cal_ego_info_in_transform_coordination(ego_dynamics, x, y, rotate_d):
    orig_x, orig_y, orig_a, corner_points = ego_dynamics['x'], ego_dynamics['y'], ego_dynamics['phi'], ego_dynamics['Corner_point']
    shifted_x, shifted_y = shift_coordination(orig_x, orig_y, x, y)
    trans_x, trans_y, trans_a = rotate_coordination(shifted_x, shifted_y, orig_a, rotate_d)
    trans_corner_points = []
    for corner_x, corner_y in corner_points:
        shifted_x, shifted_y = shift_coordination(corner_x, corner_y, x, y)
        trans_corner_x, trans_corner_y, _ = rotate_coordination(shifted_x, shifted_y, orig_a, rotate_d)
        trans_corner_points.append((trans_corner_x, trans_corner_y))
    ego_dynamics.update(dict(x=trans_x,
                             y=trans_y,
                             phi=trans_a,
                             Corner_point=trans_corner_points))
    return ego_dynamics


def xy2_edgeID_lane(x, y):
    if y < -18:
        edgeID = '1o'
        lane = 1 if x < 3.75 else 0
    elif x < -18:
        edgeID = '4i'
        lane = 1 if y < 3.75 else 0
    elif y > 18:
        edgeID = '3i'
        lane = 1 if x < 3.75 else 0
    elif x > 18:
        edgeID = '2i'
        lane = 1 if y > -3.75 else 0
    else:
        edgeID = '0'
        lane = 0
    return edgeID, lane


def _convert_car_coord_to_sumo_coord(x_in_car_coord, y_in_car_coord, a_in_car_coord, car_length):  # a in deg
    x_in_sumo_coord = x_in_car_coord + car_length / 2 * math.cos(math.radians(a_in_car_coord))
    y_in_sumo_coord = y_in_car_coord + car_length / 2 * math.sin(math.radians(a_in_car_coord))
    a_in_sumo_coord = -a_in_car_coord + 90.
    return x_in_sumo_coord, y_in_sumo_coord, a_in_sumo_coord


def _convert_sumo_coord_to_car_coord(x_in_sumo_coord, y_in_sumo_coord, a_in_sumo_coord, car_length):
    a_in_car_coord = -a_in_sumo_coord + 90.
    x_in_car_coord = x_in_sumo_coord - (math.cos(a_in_car_coord / 180. * math.pi) * car_length / 2)
    y_in_car_coord = y_in_sumo_coord - (math.sin(a_in_car_coord / 180. * math.pi) * car_length / 2)
    return x_in_car_coord, y_in_car_coord, deal_with_phi(a_in_car_coord)


def deal_with_phi(phi):
    while phi > 180:
        phi -= 360
    while phi <= -180:
        phi += 360
    return phi




class Path(object):
    """
    manage generated path
    """

    def __init__(self, percision=0.001):
        self.path = None
        self.current_step = 0
        self.total_step = None
        self.percision = percision

    def reset_path(self, dist_before_start_point, start_point_info, end_point_info, dist_after_end_point):
        self.current_step = 0
        self.path_generation(dist_before_start_point, start_point_info, end_point_info, dist_after_end_point)
        self.total_step = len(self.path)

    def path_generation(self, dist_before_start_point, start_point_info, end_point_info, dist_after_end_point):
        """
        :param dist_before_start_point:
        :param start_point_info: list, [x, y, heading(deg)]
        :param end_point_info: list, [x, y, heading(deg)]
        :param pecision: distance between points
        :return: ndarray, [[x0, y0, a0(deg)], [x1, y1, a1(deg)], ...]
        """
        # start_x, start_y, start_a = start_point_info
        # end_x, end_y, end_a = end_point_info
        # assert -180 < start_a < 180, 'start heading should be in [-180, 180](deg)'
        # assert -180 < end_a < 180, 'end heading should be in [-180, 180](deg)'
        # # transform coordination to start point
        # path = np.zeros((100, 4), dtype=np.float32)

        before_y = np.linspace(-dist_before_start_point, 0, int(dist_before_start_point / self.percision)) - 18
        before_x = 3.75 / 2 * np.ones(before_y.shape)
        before_a = 90 * np.ones(before_y.shape)
        before_points = np.array(list(zip(before_x, before_y, before_a)))

        middle_angle = np.linspace(0, np.pi / 2, int(np.pi / 2 * (18 + 3.75 / 2) / self.percision))
        middle_points = np.array(
            [[(18 + 3.75 / 2) * np.cos(a) - 18, (18 + 3.75 / 2) * np.sin(a) - 18, 90 + a * 180 / np.pi] for a in
             middle_angle[1:]])

        after_x = np.linspace(0, -dist_after_end_point, int(dist_after_end_point / self.percision)) - 18
        after_y = 3.75 / 2 * np.ones(before_y.shape)
        after_a = 180 * np.ones(before_y.shape)
        after_points = np.array(list(zip(after_x[1:], after_y[1:], after_a[1:])))

        self.path = np.vstack((before_points, middle_points, after_points))

    def get_init_state(self):
        return self.path[0]

    def get_next_info(self, delta_dist):
        delta_steps = int(np.floor(delta_dist / self.percision))
        next_index = delta_steps + self.current_step
        self.current_step = next_index if next_index < self.total_step else self.total_step - 1
        return self.path[self.current_step]

    def is_path_finished(self):
        return True if self.current_step == self.total_step - 1 else False


def test_path():
    path = Path(0.01)
    path.reset_path(5, 1, 1, 5)
    print(path.get_init_state())
    x = path.path[:, 0]
    y = path.path[:, 1]
    print(path.get_next_info(5))
    print(path.get_next_info((18 + 3.75 / 2) * np.pi / 2))
    print(path.is_path_finished())
    print(path.get_next_info(3))
    print(path.is_path_finished())
    print(path.get_next_info(3))
    print(path.is_path_finished())


if __name__ == '__main__':
    test_path()
