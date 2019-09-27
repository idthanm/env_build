import math


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

def path_generation(dist_before_start_point, start_point_info, end_point_info, dist_after_end_point, pecision):
    """
    :param dist_before_start_point:
    :param start_point_info: list, [x, y, heading(deg)]
    :param end_point_info: list, [x, y, heading(deg)]
    :param dist_after_end_point:
    :param pecision: distance between points
    :return: ndarray, [[x0, y0, a0(deg)], [x1, y1, a1(deg)], ...]
    """
    start_x, start_y, start_a = start_point_info
    end_x, end_y, end_a = end_point_info
    assert -180 < start_a < 180, 'start heading should be in [-180, 180](deg)'
    assert -180 < end_a < 180, 'end heading should be in [-180, 180](deg)'
    # transform coordination to start point


if __name__ == '__main__':
    path_generation()

