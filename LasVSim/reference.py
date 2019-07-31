import math
from LasVSim.endtoend_env_utils import shift_coordination, rotate_coordination
import copy

class Reference(object):
    def __init__(self, step_length, horizon):
        self.horizon = horizon
        self.orig_init_x = None
        self.orig_init_y = None
        self.orig_init_v = None
        self.orig_init_heading = None
        self.orig_goal_x = None
        self.orig_goal_y = None
        self.orig_goal_v = None
        self.orig_goal_heading = None  # heading in deg
        self.goal_in_ref = None
        self.goalx_in_ref = None
        self.goaly_in_ref = None
        self.goalv_in_ref = None
        self.goalheading_in_ref = None
        self.index_mode = None
        self.reference_path = None
        self.reference_velocity = None
        self.step_length = step_length  # ms
        self.sim_times = 0
        self.x = None  # in origin coordination
        self.y = None
        self.v = None
        self.heading = None
        self.orig_path_points = [] # in origin coordination
        self.horizon_path_points = []
        # self.reset_reference_path(orig_init_state, orig_goal_state)

    def reset_reference_path(self, orig_init_state, orig_goal_state):
        self.sim_times = 0
        self.orig_init_x, self.orig_init_y, self.orig_init_v, self.orig_init_heading = orig_init_state
        self.orig_goal_x, self.orig_goal_y, self.orig_goal_v, self.orig_goal_heading = orig_goal_state  # heading in deg
        self.goal_in_ref = self.orig2ref(self.orig_goal_x, self.orig_goal_y, self.orig_goal_v, self.orig_goal_heading)
        self.goalx_in_ref, self.goaly_in_ref, self.goalv_in_ref, self.goalheading_in_ref = self.goal_in_ref
        self.x = self.orig_init_x
        self.y = self.orig_init_y
        self.v = self.orig_init_v
        self.heading = self.orig_init_heading
        assert (self.goalx_in_ref > 0 and abs(self.goalheading_in_ref) < 180)
        if self.goaly_in_ref >= 0:
            assert (-90 < self.goalheading_in_ref < 180)
        else:
            assert (-180 < self.goalheading_in_ref < 90)
        self.goalheading_in_ref = self.goalheading_in_ref - 1 \
            if 90 - 0.1 < self.goalheading_in_ref < 90 + 0.1 else self.goalheading_in_ref
        self.goalheading_in_ref = self.goalheading_in_ref + 1 \
            if -90 - 0.1 < self.goalheading_in_ref < -90 + 0.1 else self.goalheading_in_ref
        self.index_mode = 'indexed_by_x' if abs(self.goalheading_in_ref) < 90 else 'indexed_by_y'
        self.reference_path = self.generate_reference_path()
        self.reference_velocity = self.generate_reference_velocity()
        self.orig_path_points = self._generate_orig_path_points()
        self.horizon_path_points = self._generate_horizon_path_points()

    def orig2ref(self, orig_x, orig_y, orig_v, orig_heading):
        orig_x, orig_y = shift_coordination(orig_x, orig_y, self.orig_init_x, self.orig_init_y)
        x_in_ref, y_in_ref, heading_in_ref = rotate_coordination(orig_x, orig_y, orig_heading, self.orig_init_heading)
        v_in_ref = orig_v
        return x_in_ref, y_in_ref, v_in_ref, heading_in_ref

    def ref2orig(self, x_in_ref, y_in_ref, v_in_ref, heading_in_ref):
        temp_x, temp_y, orig_heading = rotate_coordination(x_in_ref, y_in_ref, heading_in_ref, -self.orig_init_heading)
        orig_x, orig_y = shift_coordination(temp_x, temp_y, -self.orig_init_x, -self.orig_init_y)
        orig_v = v_in_ref
        return orig_x, orig_y, orig_v, orig_heading

    def is_pose_achieve_goal(self, orig_x, orig_y, orig_heading, distance_tolerance=1, heading_tolerance=15):
        x_in_ref, y_in_ref, _, heading_in_ref = self.orig2ref(orig_x, orig_y, 0, orig_heading)
        return True if abs(orig_x - self.goalx_in_ref) < distance_tolerance and \
                       abs(orig_y - self.goaly_in_ref) < distance_tolerance and \
                       abs(orig_heading - self.goalheading_in_ref) < heading_tolerance else False

    def is_legit(self, orig_x, orig_y):
        x_in_ref, y_in_ref, v_in_ref, heading_in_ref = self.orig2ref(orig_x, orig_y, 0, 0)
        return True if 0 < x_in_ref < self.goalx_in_ref or abs(y_in_ref) < abs(self.goaly_in_ref) else False

    def cal_bias(self, orig_x, orig_y, orig_v, orig_heading):
        assert self.index_mode == 'indexed_by_x' or self.index_mode == 'indexed_by_y'
        x_in_ref, y_in_ref, v_in_ref, heading_in_ref = self.orig2ref(orig_x, orig_y, orig_v, orig_heading)
        if self.index_mode == 'indexed_by_x':
            refer_point = self.access_path_point_indexed_by_x(
                x_in_ref) if x_in_ref < self.goalx_in_ref else self.goal_in_ref
        else:
            refer_point = self.access_path_point_indexed_by_y(
                y_in_ref) if y_in_ref < self.goaly_in_ref else self.goal_in_ref
        ref_x, ref_y, _, ref_heading = refer_point  # for now, we use goalv_in_ref instead ref_v
        position_bias = math.sqrt((ref_x - x_in_ref) ** 2 + (ref_y - y_in_ref) ** 2)
        velocity_bias = abs(self.goalv_in_ref - v_in_ref)
        heading_bias = abs(ref_heading - heading_in_ref)
        return position_bias, velocity_bias, heading_bias

    def generate_reference_path(self):  # for now, path is three order poly
        reference_path = []  # list of coefficient
        assert self.index_mode == 'indexed_by_x' or self.index_mode == 'indexed_by_y'
        if self.index_mode == 'indexed_by_x':
            a0 = 0
            a1 = 0
            slope = self._deg2slope(self.goalheading_in_ref)
            a3 = (slope - 2 * self.goaly_in_ref / self.goalx_in_ref) / self.goalx_in_ref ** 2
            a2 = self.goaly_in_ref / self.goalx_in_ref ** 2 - a3 * self.goalx_in_ref
            reference_path = [a0, a1, a2, a3]
        else:
            trans_x, trans_y, trans_d = rotate_coordination(self.goalx_in_ref, self.goaly_in_ref,
                                                            self.goalheading_in_ref, -90)
            a0 = 0
            a1 = -math.tan((90 - 1) * math.pi / 180) if trans_x <= 0 else math.tan((90 - 1) * math.pi / 180)
            a3 = (self._deg2slope(trans_d) - a1 - 2 * (trans_y - a1 * trans_x) / trans_x) / trans_x ** 2
            a2 = (trans_y - a1 * trans_x - a3 * trans_x ** 3) / trans_x ** 2
            reference_path = [a0, a1, a2, a3]
        return reference_path

    def generate_reference_velocity(self):  # for now, path is linear function
        reference_velocity = []  # list of weight (no bias)
        assert self.index_mode == 'indexed_by_x' or self.index_mode == 'indexed_by_y'
        if self.index_mode == 'indexed_by_x':
            reference_velocity.append((self.goalv_in_ref - self.orig_init_v) / self.goalx_in_ref)
        else:
            reference_velocity.append((self.goalv_in_ref - self.orig_init_v) / self.goalx_in_ref)
        return reference_velocity

    def access_path_point_indexed_by_x(self, x_in_ref):
        assert (self.index_mode == 'indexed_by_x')
        assert (0 < x_in_ref < self.goalx_in_ref)
        a0, a1, a2, a3 = self.reference_path
        w = self.reference_velocity[0]
        y_in_ref = a0 + a1 * x_in_ref + a2 * x_in_ref ** 2 + a3 * x_in_ref ** 3
        v = w * x_in_ref + self.orig_init_v
        slope = a1 + 2 * a2 * x_in_ref + 3 * a3 * x_in_ref ** 2
        heading_in_ref = self._slope2deg(slope)
        return x_in_ref, y_in_ref, v, heading_in_ref

    def access_path_point_indexed_by_y(self, y_in_ref):
        assert (self.index_mode == 'indexed_by_y')
        a0, a1, a2, a3 = self.reference_path
        w = self.reference_velocity[0]
        x_in_ref = a0 + a1 * (-y_in_ref) + a2 * (-y_in_ref) ** 2 + a3 * (-y_in_ref) ** 3
        v = w * abs(y_in_ref)
        slope = a1 + 2 * a2 * (-y_in_ref) + 3 * a3 * (-y_in_ref) ** 2
        temp_heading = self._slope2deg(slope)
        if self.goaly_in_ref > 0:
            heading_in_ref = temp_heading + 90
        else:
            heading_in_ref = temp_heading - 90
        return x_in_ref, y_in_ref, v, heading_in_ref

    def _deg2slope(self, deg):
        return math.tan(deg * math.pi / 180)

    def _slope2deg(self, slope):
        return 180 * math.atan(slope) / math.pi

    def sim_step(self):
        if self.sim_times < len(self.orig_path_points):
            self.x = self.orig_path_points[self.sim_times]['x']
            self.y = self.orig_path_points[self.sim_times]['y']
            self.v = self.orig_path_points[self.sim_times]['v']
            self.heading = self.orig_path_points[self.sim_times]['heading']
            self.sim_times += 1
            return self.x, self.y, self.v, self.heading
        else:
            self.x += self.v * self.step_length/1000
            self.heading = 0
            self.sim_times += 1
            return self.x, self.y, self.v, self.heading

    def _generate_orig_path_points(self):  # not including initial point
        orig_path_points = []
        if self.index_mode == 'indexed_by_x':
            x_in_ref, y_in_ref, v_x, heading_in_ref = 0, 0, self.v, 0
            x_in_ref += v_x * self.step_length / 1000.0
            while x_in_ref < self.goalx_in_ref and len(orig_path_points) < self.horizon:
                x_in_ref, y_in_ref, v, heading_in_ref = self.access_path_point_indexed_by_x(x_in_ref)
                v_x = v * math.cos(self._deg2slope(heading_in_ref))
                orig_x, orig_y, orig_v, orig_heading = self.ref2orig(x_in_ref, y_in_ref, v, heading_in_ref)
                orig_path_points.append(dict(x=orig_x,
                                             y=orig_y,
                                             v=orig_v,
                                             heading=orig_heading
                                             ))
                x_in_ref += v_x * self.step_length / 1000.0
        return orig_path_points

    def _generate_horizon_path_points(self):
        horizon_path_points = []
        if len(self.orig_path_points) >= self.horizon:
            horizon_path_points = self.orig_path_points[0:self.horizon]
        else:
            horizon_path_points = copy.deepcopy(self.orig_path_points)
            x = horizon_path_points[-1]['x']
            y = horizon_path_points[-1]['y']
            v = horizon_path_points[-1]['v']
            heading = horizon_path_points[-1]['heading']
            while len(horizon_path_points) < self.horizon:
                x = x + v * self.step_length/1000
                horizon_path_points.append(dict(x=x,
                                                y=y,
                                                v=v,
                                                heading=0
                                                ))
        assert len(horizon_path_points) == self.horizon
        return horizon_path_points


