import numpy as np
import gym
from scipy.optimize import minimize
import time


# class VehicleDynamics(object):
#     def __init__(self, ):
#         self.vehicle_params = dict(C_f=88000.,  # front wheel cornering stiffness [N/rad]
#                                    C_r=94000.,  # rear wheel cornering stiffness [N/rad]
#                                    a=1.14,  # distance from CG to front axle [m]
#                                    b=1.40,  # distance from CG to rear axle [m]
#                                    mass=1500.,  # mass [kg]
#                                    I_z=2420.,  # Polar moment of inertia at CG [kg*m^2]
#                                    miu=1.0,  # tire-road friction coefficient
#                                    g=9.81,  # acceleration of gravity [m/s^2]
#                                    )
#         a, b, mass, g = self.vehicle_params['a'], self.vehicle_params['b'], \
#                         self.vehicle_params['mass'], self.vehicle_params['g']
#         F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
#         self.vehicle_params.update(dict(F_zf=F_zf,
#                                         F_zr=F_zr))
#
#     def f_xu(self, states, actions):  # states and actions are tensors, [[], [], ...]
#         v_x, v_y, r, x, y, phi = states
#         phi = phi * np.pi / 180.
#         steer, a_x = actions
#         C_f = self.vehicle_params['C_f']
#         C_r = self.vehicle_params['C_r']
#         a = self.vehicle_params['a']
#         b = self.vehicle_params['b']
#         mass = self.vehicle_params['mass']
#         I_z = self.vehicle_params['I_z']
#         miu = self.vehicle_params['miu']
#         g = self.vehicle_params['g']
#
#         F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
#         F_xf = mass * a_x / 2 if a_x<0 else 0
#         F_xr = mass * a_x / 2 if a_x<0 else mass * a_x
#         miu_f = np.sqrt(np.square(miu * F_zf) - np.square(F_xf)) / F_zf
#         miu_r = np.sqrt(np.square(miu * F_zr) - np.square(F_xr)) / F_zr
#         alpha_f = np.arctan((v_y + a * r) / v_x) - steer
#         alpha_r = np.arctan((v_y - b * r) / v_x)
#
#         Ff_w1 = np.square(C_f) / (3 * F_zf * miu_f)
#         Ff_w2 = np.power(C_f, 3) / (27 * np.power(F_zf * miu_f, 2))
#         F_yf_max = F_zf * miu_f
#
#         Fr_w1 = np.square(C_r) / (3 * F_zr * miu_r)
#         Fr_w2 = np.power(C_r, 3) / (27 * np.power(F_zr * miu_r, 2))
#         F_yr_max = F_zr * miu_r
#
#         F_yf = - C_f * np.tan(alpha_f) + Ff_w1 * np.tan(alpha_f) * np.abs(
#             np.tan(alpha_f)) - Ff_w2 * np.power(np.tan(alpha_f), 3)
#         F_yr = - C_r * np.tan(alpha_r) + Fr_w1 * np.tan(alpha_r) * np.abs(
#             np.tan(alpha_r)) - Fr_w2 * np.power(np.tan(alpha_r), 3)
#
#         F_yf = np.minimum(F_yf, F_yf_max)
#         F_yf = np.minimum(F_yf, -F_yf_max)
#
#         F_yr = np.minimum(F_yr, F_yr_max)
#         F_yr = np.minimum(F_yr, -F_yr_max)
#
#         state_deriv = [a_x + v_y * r,
#                        (F_yf * np.cos(steer) + F_yr) / mass - v_x * r,
#                        (a * F_yf * np.cos(steer) - b * F_yr) / I_z,
#                        v_x * np.cos(phi) - v_y * np.sin(phi),
#                        v_x * np.sin(phi) + v_y * np.cos(phi),
#                        r * 180 / np.pi,
#                        ]
#
#         state_deriv_stack = np.array(state_deriv)
#
#         return state_deriv_stack
#
#     def prediction(self, x_1, u_1, frequency, RK):
#         f_xu_1 = self.f_xu(x_1, u_1)
#         x_next = f_xu_1 / frequency + x_1
#         return x_next

class VehicleDynamics(object):
    def __init__(self, ):
        self.vehicle_params = dict(C_f=88000.,  # front wheel cornering stiffness [N/rad]
                                   C_r=94000.,  # rear wheel cornering stiffness [N/rad]
                                   a=1.14,  # distance from CG to front axle [m]
                                   b=1.40,  # distance from CG to rear axle [m]
                                   mass=1500.,  # mass [kg]
                                   I_z=2420.,  # Polar moment of inertia at CG [kg*m^2]
                                   miu=1.0,  # tire-road friction coefficient
                                   g=9.81,  # acceleration of gravity [m/s^2]
                                   )
        a, b, mass, g = self.vehicle_params['a'], self.vehicle_params['b'], \
                        self.vehicle_params['mass'], self.vehicle_params['g']
        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        self.vehicle_params.update(dict(F_zf=F_zf,
                                        F_zr=F_zr))

    def f_xu(self, states, actions, tau):  # states and actions are tensors, [[], [], ...]
        v_x, v_y, r, x, y, phi = states
        phi = phi * np.pi / 180.
        steer, a_x = actions
        C_f = self.vehicle_params['C_f']
        C_r = self.vehicle_params['C_r']
        a = self.vehicle_params['a']
        b = self.vehicle_params['b']
        mass = self.vehicle_params['mass']
        I_z = self.vehicle_params['I_z']
        miu = self.vehicle_params['miu']
        g = self.vehicle_params['g']

        next_state = [v_x + tau*(a_x + v_y * r),
                      (mass*v_y*v_x+tau*(a*C_f-b*C_r)*r-tau*C_f*steer*v_x-tau*mass*np.square(v_x)*r)/(mass*v_x-tau*(C_f+C_r)),
                      (-I_z*r*v_x-tau*(a*C_f-b*C_r)*v_y+tau*a*C_f*steer*v_x)/(tau*(a**2*C_f+b**2*C_r)-I_z*v_x),
                      x+tau*(v_x * np.cos(phi) - v_y * np.sin(phi)),
                      y+tau*(v_x * np.sin(phi) + v_y * np.cos(phi)),
                      (phi+tau*r) * 180 / np.pi]

        return np.array(next_state)

    def prediction(self, x_1, u_1, frequency, RK):
        x_next = self.f_xu(x_1, u_1, 1/frequency)
        return x_next


class ModelPredictiveControl:
    def __init__(self, init_x, horizon):
        self.fre = 10
        self.horizon = horizon
        self.init_x = init_x
        self.vehicle_dynamics = VehicleDynamics()
        self.task = 'left'
        self.exp_v = 10.
        self.path = None

    def reset_init_x(self, init_x, path):
        self.init_x = init_x
        self.path = path

    def plant_model(self, u, x):
        x_copy = x.copy()
        x_copy = self.vehicle_dynamics.prediction(x_copy[:6], u, self.fre, 1)
        return x_copy

    def cost_function(self, u):
        u = u.reshape(self.horizon, 2)
        loss = 0.
        x = self.init_x.copy()
        for i in range(0, self.horizon):
            x = self.plant_model(u[i], x)
            v_x, v_y, r, ego_x, ego_y, phi = x[:6]
            dists = np.square(self.path[0] - ego_x) + np.square(self.path[1] - ego_y)
            ref_index = np.argmin(dists)
            ref_x, ref_y, ref_phi = self.path[0][ref_index], \
                                    self.path[1][ref_index], \
                                    self.path[2][ref_index]
            loss += 0.01*np.square(v_x - self.exp_v)
            loss += 0.1*(np.square(ego_x - ref_x) + np.square(ego_y - ref_y))
            loss += 5*np.square((phi - ref_phi) * np.pi / 180.)

        return loss


if __name__ == '__main__':
    horizon_list = [10]
    env = gym.make('CrossroadEnd2end-v0', training_task='left', num_future_data=0)
    done = 0
    for horizon in horizon_list:
        for i in range(10):
            obs = env.reset()
            mpc = ModelPredictiveControl(obs, horizon)
            bounds = [(-0.2, 0.2), (-3., 3.)] * horizon
            u_init = np.zeros((horizon, 2))
            mpc.reset_init_x(obs, env.ref_path.path)

            while not done:
                start_time = time.time()
                results = minimize(mpc.cost_function,
                                  x0=u_init.flatten(),
                                  method='SLSQP',
                                  bounds=bounds,
                                  tol=1e-1)
                action = results.x
                print(action)
                print(results.success, results.message)
                end_time = time.time()

                u_init = np.concatenate([action[2:], action[-2:]])
                obs, reward, done, info = env.step(action[:2])
                mpc.reset_init_x(obs, env.ref_path.path)
                env.render()
            done = 0





