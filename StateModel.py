from __future__ import print_function
import torch
import numpy as np
from Config import StateConfig
import random
import math
import matplotlib.pyplot as plt


class StateModel(StateConfig):
    def __init__(self):
        super(StateModel, self).__init__()

    def Trajectory(self, x, y):

        lane_position = 15 * np.cos(x / 70) + 5 * np.cos(x / 95) - 10 * np.cos(x / 110)
        lane_angle = np.arctan(-15 / 70 * np.sin(x / 70) - 5 / 95 * np.sin(x / 95) + 10 / 110 * np.sin(x / 110))

        '''
        b = x // (2 * self.radius)
        a = x % (2 * self.radius)
        c = np.where(b % 2 == 0, 1.0, -1.0)
        a[abs(a) < 1e-4] = 1e-4
        a[abs(2*self.radius-a) < 1e-4] = 2*self.radius-1e-4
        dist_to_c = self.radius - a
        lane_position = c * np.sqrt(np.square(self.radius) - np.square(dist_to_c))
        lane_angle = np.arctan(dist_to_c / lane_position)
        '''

        return lane_position, lane_angle

    def StateFunction(self, inputs, u):

        v_long = inputs[:, 0]
        v_lat = inputs[:, 1]
        v_ang = inputs[:, 2]
        head_ang = inputs[:, 3]
        delta_y = inputs[:, 4]

        steer_ang = u[:, 0]
        acc = u[:, 1]

        acc_f = torch.where(acc < 0, acc / 2, torch.zeros(len(acc)))
        acc_r = torch.where(acc < 0, acc / 2, acc)

        F_xr = self.m * acc_r
        F_xf = self.m * acc_f

        u_r = torch.div(torch.sqrt(pow(self.u_tire * self.Fzr, 2) - torch.pow(F_xr, 2)), self.Fzr)
        u_f = torch.div(torch.sqrt(pow(self.u_tire * self.Fzf, 2) - torch.pow(F_xf, 2)), self.Fzf)

        Ff_w1 = pow(self.Caf, 2) / (3 * self.Fzf * u_f)
        Ff_w2 = pow(self.Caf, 3) / (27 * torch.pow(self.Fzf * u_f, 2))
        self.F_yf_max = self.Fzf * u_f

        Fr_w1 = pow(self.Car, 2) / (3 * self.Fzr * u_r)
        Fr_w2 = pow(self.Car, 3) / (27 * torch.pow(self.Fzr * u_r, 2))
        self.F_yr_max = self.Fzr * u_r

        alpha_f = torch.atan(torch.div(v_lat + self.a * v_ang, v_long)) - steer_ang
        alpha_r = torch.atan(torch.div(v_lat - self.b * v_ang, v_long))

        F_yf = - self.Caf * torch.tan(alpha_f) + torch.mul(Ff_w1, torch.mul(torch.tan(alpha_f),
                                                                            torch.abs(torch.tan(alpha_f)))) - torch.mul(
            Ff_w2, torch.pow(torch.tan(alpha_f), 3))
        F_yr = - self.Car * torch.tan(alpha_r) + torch.mul(Fr_w1, torch.mul(torch.tan(alpha_r),
                                                                            torch.abs(torch.tan(alpha_r)))) - torch.mul(
            Fr_w2, torch.pow(torch.tan(alpha_r), 3))

        F_yf = torch.min(F_yf, self.F_yf_max)
        F_yf = torch.max(F_yf, -self.F_yf_max)

        F_yr = torch.min(F_yr, self.F_yr_max)
        F_yr = torch.max(F_yr, -self.F_yr_max)

        deri_v_long = acc + torch.mul(v_lat, v_ang)  # - torch.div(torch.mul(F_yf, torch.sin(steer_ang)), self.m)
        deri_v_lat = torch.div(torch.mul(F_yf, torch.cos(steer_ang)) + F_yr, self.m) - torch.mul(v_long, v_ang)
        deri_v_ang = torch.div(torch.mul(torch.mul(F_yf, self.a), torch.cos(steer_ang)) - torch.mul(F_yr, self.b),
                               self.Iz)
        deri_head_ang = v_ang  # - torch.div(torch.mul(v_long, torch.cos(head_ang)) - torch.mul(v_lat, torch.sin(head_ang)),self.radius-delta_y)
        deri_position_y = torch.mul(v_long, torch.sin(head_ang)) + torch.mul(v_lat, torch.cos(head_ang))

        f_xu = torch.cat((deri_v_long[np.newaxis, :], deri_v_lat[np.newaxis, :], deri_v_ang[np.newaxis, :],
                          deri_head_ang[np.newaxis, :], deri_position_y[np.newaxis, :]), 0).t()

        return f_xu, F_yf, F_yr, alpha_f, alpha_r, u_f, u_r

    def CostFunction(self, inputs, u):
        v_long = inputs[:, 0]
        v_lat = inputs[:, 1]
        v_ang = inputs[:, 2]
        head_ang = inputs[:, 3]

        position_y = inputs[:, 4]
        # position_x = inputs[:,5]

        steer_ang = u[:, 0]
        acc = u[:, 1]

        # L = 0.0005 * (-30 * v_long + 80 * torch.pow(position_y, 2) + 200 * torch.pow(steer_ang, 2) + 0.24 * torch.pow(acc, 2))
        L = 0.0005 * (-30 * v_long + 160 * torch.pow(position_y, 2) + 200 * torch.pow(steer_ang, 2) + 0.24 * torch.pow(
            acc, 2))

        return 1 * L[np.newaxis, :]

    def Hamilton(self, delta_value, L, f_xu):

        dv_t = torch.diag(torch.mm(delta_value, f_xu), 0)
        hamilton = L + dv_t
        lyapunov = dv_t
        return hamilton, lyapunov

    def Prediction(self, x_1, u_1, frequency, RK):
        if RK == 1:
            f_xu_1, F_yf, F_yr, a_f, a_r, u_f, u_r = self.StateFunction(x_1, u_1)
            x_next = f_xu_1 / frequency + x_1

        elif RK == 2:
            f_xu_1, F_yf, F_yr, a_f, a_r, u_f, u_r = self.StateFunction(x_1, u_1)
            K1 = (1 / frequency) * f_xu_1
            x_2 = x_1 + K1
            # u_2 = policynet(x_2)
            f_xu_2, _, _, _, _, _ = self.StateFunction(x_2, u_1)
            K2 = (1 / frequency) * f_xu_2
            x_next = x_1 + (K1 + K2) / 2
        elif RK == 4:
            f_xu_1, F_yf, F_yr, a_f, a_r, u_f, u_r = self.StateFunction(x_1, u_1)
            K1 = (1 / frequency) * f_xu_1
            x_2 = x_1 + K1 / 2
            # u_2 = policynet(x_2)
            f_xu_2, _, _, _, _, _, _ = self.StateFunction(x_2, u_1)
            K2 = (1 / frequency) * f_xu_2
            x_3 = x_1 + K2 / 2
            f_xu_3, _, _, _, _, _, _ = self.StateFunction(x_3, u_1)
            K3 = (1 / frequency) * f_xu_3
            x_4 = x_1 + K3
            f_xu_4, _, _, _, _, _, _ = self.StateFunction(x_4, u_1)
            K4 = (1 / frequency) * f_xu_4
            x_next = x_1 + (K1 + 2 * K2 + 2 * K3 + K4) / 6

        return x_next, F_yf, F_yr, a_f, a_r, u_f, u_r

    def Simulation(self, x_state_, x_agent_, u):
        for i in range(self.frequency_time):
            x = torch.from_numpy(x_state_.copy()).float()
            x, F_yf, F_yr, a_f, a_r, u_f, u_r = self.Prediction(x.detach(), u.detach(), self.frequency_simulation, 1)
            # print(x.requires_grad)
            x_state_ = x.detach().numpy().copy()
            v_long = x_agent_[:, 0]
            v_lat = x_agent_[:, 1]
            v_ang = x_agent_[:, 2]
            head_ang = x_agent_[:, 3]

            x_state_[:, 0][x_state_[:, 0] > StateConfig.v_max] = StateConfig.v_max
            x_state_[:, 0][x_state_[:, 0] < 1] = 1

            x_agent_[:, 3] += v_ang / self.frequency_simulation
            x_agent_[:, 4] += (v_long * np.sin(head_ang) + v_lat * np.cos(head_ang)) / self.frequency_simulation
            x_agent_[:, 5] += (v_long * np.cos(head_ang) - v_lat * np.sin(head_ang)) / self.frequency_simulation
            x_agent_[:, 0:3] = x_state_[:, 0:3].copy()

            lane_position, lane_angle = self.Trajectory(x_agent_[:, -1], x_agent_[:, -2])

            x_agent_[:, 3][x_agent_[:, 3] > math.pi] -= 2 * math.pi
            x_agent_[:, 3][x_agent_[:, 3] <= -math.pi] += 2 * math.pi
            x_state_[:, 3] = x_agent_[:, 3] - lane_angle
            x_state_[:, 4] = x_agent_[:, 4] - lane_position
            x_state_[:, 3][x_state_[:, 3] > math.pi] -= 2 * math.pi
            x_state_[:, 3][x_state_[:, 3] <= -math.pi] += 2 * math.pi

        return x_state_, x_agent_, F_yf, F_yr, a_f, a_r, u_f, u_r


def test():
    aaa = np.array([0.0000001, -0.00000001, 0, -1, 2])
    aaa[abs(aaa) < 1e-4] = 1e-4
    print(aaa)
    x = np.arange(0, 1000, 0.5)
    lane_position = 15 * np.cos(x / 50) + 5 * np.cos(x / 65) - 10 * np.cos(x / 90)
    lane_angle = np.arctan(-15 / 50 * np.sin(x / 50) - 5 / 65 * np.sin(x / 85) + 10 / 90 * np.sin(x / 90))

    fig = plt.figure(0)
    # fig.add_axes([0.08, 0.08, 0.98, 0.98])
    plt.subplot(211)
    plt.plot(x, lane_position)
    plt.subplot(212)
    plt.plot(x, lane_angle)
    plt.show()


if __name__ == "__main__":
    test()
