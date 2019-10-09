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
    transformed_x, transformed_y, transformed_d(range:[-180 deg, 180 deg])
    """

    coordi_rotate_d_in_rad = coordi_rotate_d * math.pi / 180
    transformed_x = orig_x * math.cos(coordi_rotate_d_in_rad) + orig_y * math.sin(coordi_rotate_d_in_rad)
    transformed_y = -orig_x * math.sin(coordi_rotate_d_in_rad) + orig_y * math.cos(coordi_rotate_d_in_rad)
    transformed_d = orig_d - coordi_rotate_d
    if transformed_d > 180:
        transformed_d = transformed_d - 360
    elif transformed_d < -180:
        transformed_d = transformed_d + 360
    else:
        transformed_d = transformed_d
    return transformed_x, transformed_y, transformed_d


def shift_and_rotate_coordination(orig_x, orig_y, orig_d, coordi_shift_x, coordi_shift_y, coordi_rotate_d):
    shift_x, shift_y = shift_coordination(orig_x, orig_y, coordi_shift_x, coordi_shift_y)
    transformed_x, transformed_y, transformed_d \
        = rotate_coordination(shift_x, shift_y, orig_d, coordi_rotate_d)
    return transformed_x, transformed_y, transformed_d


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
        before_y = np.linspace(-dist_before_start_point, 0, int(dist_before_start_point/self.percision))-18
        before_x = 3.75/2*np.ones(before_y.shape)
        before_a = 90 * np.ones(before_y.shape)
        before_points = np.array(list(zip(before_x, before_y, before_a)))

        middle_angle = np.linspace(0, np.pi/2, int(np.pi/2*(18+3.75/2)/self.percision))
        middle_points = np.array([[(18+3.75/2)*np.cos(a)-18, (18+3.75/2)*np.sin(a)-18, 90+a*180/np.pi] for a in middle_angle[1:]])

        after_x = np.linspace(0, -dist_after_end_point, int(dist_after_end_point / self.percision)) - 18
        after_y = 3.75/2 * np.ones(before_y.shape)
        after_a = 180 * np.ones(before_y.shape)
        after_points = np.array(list(zip(after_x[1:], after_y[1:], after_a[1:])))

        self.path = np.vstack((before_points, middle_points, after_points))

    def get_init_state(self):
        return self.path[0]

    def get_next_info(self, delta_dist):
        delta_steps = int(np.floor(delta_dist/self.percision))
        next_index = delta_steps + self.current_step
        self.current_step = next_index if next_index < self.total_step else self.total_step - 1
        return self.path[self.current_step]

    def is_path_finished(self):
        return True if self.current_step == self.total_step - 1 else False


def test_path():
    import matplotlib.pyplot as plt
    path = Path(0.01)
    path.reset_path(5,1,1,5)
    print(path.get_init_state())
    x = path.path[:, 0]
    y = path.path[:, 1]
    print(path.get_next_info(5))
    print(path.get_next_info((18+3.75/2)*np.pi/2))
    print(path.is_path_finished())
    print(path.get_next_info(3))
    print(path.is_path_finished())
    print(path.get_next_info(3))
    print(path.is_path_finished())


if __name__ == '__main__':
    test_path()
