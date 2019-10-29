import random
import numpy as np
from math import pi, hypot, sin, cos, asin, tan

V_max = 30/3.6; V_min = 20/3.6 # (m/s)
a_max = 1; a_min = -1 # (m^2/s)
delta_max = pi / 6; delta_min = - pi / 6 # limit of delta
T = 0.1 # sample time

epsilon = 1 # threshold of the target check
regionScale = 0.5 # start position random region

C = 2.7 # distance form rear to forward
L = 4.7 # length of the vehicle
W = 2.0 # width of the vehicle
offset = 1.35 # distance from rear to center

latDist = 0.2
longDist = 0.2
THW = 0.1

class Vehicle(object):
    def __init__(self, trafficModel, manualDist = 0):

        self.trafficModel = trafficModel

        if trafficModel.flag[0] == 'D':
            self.posx = trafficModel.start[0]
            self.posy = trafficModel.start[1] + regionScale * abs(trafficModel.start[1]) * random.random() + manualDist
            self.theta = 0.5 * pi

        if trafficModel.flag[0] == 'R':
            self.posx = (1 - regionScale) * trafficModel.start[0] + regionScale * abs(trafficModel.start[0]) * random.random()
            self.posy = trafficModel.start[1]
            self.theta = pi

        if trafficModel.flag[0] == 'U':
            self.posx = trafficModel.start[0]
            self.posy = (1 - regionScale) * trafficModel.start[1] + regionScale * abs(trafficModel.start[1]) * random.random()
            self.theta = - 0.5 * pi

        if trafficModel.flag[0] == 'L':
            self.posx = trafficModel.start[0] + regionScale * abs(trafficModel.start[0]) * random.random()
            self.posy = trafficModel.start[1]
            self.theta = 0

        self.target = trafficModel.end # self.target is a tuple i.e. (x, y)
        self.vel = V_min + (V_max - V_min) * random.random()
        # self.vel = V_min
        self.C = C # distance form rear to forward
        self.L = L # length of the vehicle
        self.W = W # width of the vehicle
        self.offset = offset # distance from rear to center
        self.R = hypot(self.L, self.W) / 2.0 # radius of the vehicle

        self.delta = 0
        self.cen_x, self.cen_y = self.cenPos()
        self.boundX, self.boundY = self.carBox()
        self.safeX, self.safeY = self.carSafeBox()

        self.history = [] # self.history.append((x, y, theta))

        self.R_max = self.C / tan(delta_max)
        self.alpha = 0
        self.endFlag = False

        if trafficModel.flag == 'DR':#1
            self.ref = [(trafficModel.start[0],trafficModel.start[1]),
                        (trafficModel.start[0], trafficModel.end[1]-self.R_max),
                        (trafficModel.start[0]+self.R_max, trafficModel.end[1]),
                        (trafficModel.end[0],trafficModel.end[1])]
            self.middlePoint = (self.ref[2][0] - self.R_max * cos(0.25 * pi),
                                self.ref[1][1] + self.R_max * cos(0.25 * pi))

        elif trafficModel.flag == 'DU':#2
            self.ref = [(trafficModel.start[0], trafficModel.start[1]),
                        (trafficModel.start[0], trafficModel.start[1] + self.R_max),
                        (trafficModel.start[0], trafficModel.end[1]- self.R_max),
                        (trafficModel.start[0], trafficModel.end[1])]
            self.middlePoint = (self.ref[0][0], 0)

        elif trafficModel.flag == 'DL':#3
            self.ref = [(trafficModel.start[0], trafficModel.start[1]),
                        (trafficModel.start[0], trafficModel.end[1] - self.R_max),
                        (trafficModel.start[0] - self.R_max, trafficModel.end[1]),
                        (trafficModel.end[0], trafficModel.end[1])]
            self.middlePoint = (self.ref[2][0] + self.R_max * cos(0.25 * pi),
                                self.ref[1][1] + self.R_max * cos(0.25 * pi))

        elif trafficModel.flag == 'RU':#4
            self.ref = [(trafficModel.start[0], trafficModel.start[1]),
                        (trafficModel.end[0] +  self.R_max, trafficModel.start[1]),
                        (trafficModel.end[0], trafficModel.start[1] +  self.R_max),
                        (trafficModel.end[0], trafficModel.end[1])]
            self.middlePoint = (self.ref[1][0] - self.R_max * cos(0.25 * pi),
                                self.ref[2][1] - self.R_max * cos(0.25 * pi))

        elif trafficModel.flag == 'RL':#5
            self.ref = [(trafficModel.start[0], trafficModel.start[1]),
                        (trafficModel.start[0] - self.R_max, trafficModel.end[1] ),
                        (trafficModel.end[0] + self.R_max, trafficModel.end[1]),
                        (trafficModel.end[0], trafficModel.end[1])]
            self.middlePoint = (0, self.ref[0][1])

        elif trafficModel.flag == 'RD':#6
            self.ref = [(trafficModel.start[0], trafficModel.start[1]),
                        (trafficModel.end[0] + self.R_max, trafficModel.start[1] ),
                        (trafficModel.end[0] , trafficModel.start[1]- self.R_max),
                        (trafficModel.end[0], trafficModel.end[1])]
            self.middlePoint = (self.ref[1][0] - self.R_max * cos(0.25 * pi),
                                self.ref[2][1] + self.R_max * cos(0.25 * pi))

        elif trafficModel.flag == 'LD':#7
            self.ref = [(trafficModel.start[0], trafficModel.start[1]),
                        (trafficModel.end[0] - self.R_max, trafficModel.start[1] ),
                        (trafficModel.end[0] , trafficModel.start[1]- self.R_max),
                        (trafficModel.end[0], trafficModel.end[1])]
            self.middlePoint = (self.ref[1][0] + self.R_max * cos(0.25 * pi),
                                self.ref[2][1] + self.R_max * cos(0.25 * pi))

        elif trafficModel.flag == 'LR':  # 8
            self.ref = [(trafficModel.start[0], trafficModel.start[1]),
                        (trafficModel.start[0] + self.R_max, trafficModel.end[1]),
                        (trafficModel.end[0] - self.R_max, trafficModel.end[1]),
                        (trafficModel.end[0], trafficModel.end[1])]
            self.middlePoint = (0, self.ref[0][1])

        elif trafficModel.flag == 'LU':  # 9
            self.ref = [(trafficModel.start[0], trafficModel.start[1]),
                        (trafficModel.end[0] - self.R_max, trafficModel.start[1]),
                        (trafficModel.end[0] , trafficModel.start[1]+ self.R_max),
                        (trafficModel.end[0], trafficModel.end[1])]
            self.middlePoint = (self.ref[1][0] + self.R_max * cos(0.25 * pi),
                                self.ref[2][1] - self.R_max * cos(0.25 * pi))


        elif trafficModel.flag == 'UL':  # 10
            self.ref = [(trafficModel.start[0], trafficModel.start[1]),
                        (trafficModel.start[0] , trafficModel.end[1]+ self.R_max),
                        (trafficModel.start[0] - self.R_max, trafficModel.end[1]),
                        (trafficModel.end[0], trafficModel.end[1])]
            self.middlePoint = (self.ref[2][0] + self.R_max * cos(0.25 * pi),
                                self.ref[1][1] - self.R_max * cos(0.25 * pi))

        elif trafficModel.flag == 'UD':  # 11
            self.ref = [(trafficModel.start[0], trafficModel.start[1]),
                        (trafficModel.start[0] , trafficModel.start[1]- self.R_max),
                        (trafficModel.end[0] , trafficModel.end[1]+ self.R_max),
                        (trafficModel.end[0], trafficModel.end[1])]
            self.middlePoint = (self.ref[1][0], 0)

        else:  # 12
            self.ref = [(trafficModel.start[0], trafficModel.start[1]),
                        (trafficModel.start[0] , trafficModel.end[1]+ self.R_max),
                        (trafficModel.start[0] + self.R_max, trafficModel.end[1]),
                        (trafficModel.end[0], trafficModel.end[1])]
            self.middlePoint = (self.ref[2][0] - self.R_max * cos(0.25 * pi),
                                self.ref[1][1] - self.R_max * cos(0.25 * pi))


    def carBox(self):
        x0 = np.mat([[self.cen_x], [self.cen_y]])
        car1 = x0[0:2] + np.mat([[np.cos(self.theta) * self.L / 2], [np.sin(self.theta) * self.L / 2]]) + np.mat(
            [[np.sin(self.theta) * self.W / 2], [-np.cos(self.theta) * self.W / 2]])
        car2 = x0[0:2] + np.mat([[np.cos(self.theta) * self.L / 2], [np.sin(self.theta) * self.L / 2]]) - np.mat(
            [[np.sin(self.theta) * self.W / 2], [-np.cos(self.theta) * self.W / 2]])
        car3 = x0[0:2] - np.mat([[np.cos(self.theta) * self.L / 2], [np.sin(self.theta) * self.L / 2]]) + np.mat(
            [[np.sin(self.theta) * self.W / 2], [-np.cos(self.theta) * self.W / 2]])
        car4 = x0[0:2] - np.mat([[np.cos(self.theta) * self.L / 2], [np.sin(self.theta) * self.L / 2]]) - np.mat(
            [[np.sin(self.theta) * self.W / 2], [-np.cos(self.theta) * self.W / 2]])
        x = [car1[0, 0], car2[0, 0], car4[0, 0], car3[0, 0], car1[0, 0]]
        y = [car1[1, 0], car2[1, 0], car4[1, 0], car3[1, 0], car1[1, 0]]
        return x, y


    def carSafeBox(self):
        # if self.theta == 0 or 0.5 * pi or pi or (-0.5 * pi) or (- pi):
        '''if self.delta == 0:
            forwardDist = longDist + self.vel * THW
        else:
            forwardDist = longDist

        car1 = np.mat([[self.boundX[0]], [self.boundY[0]]]) + \
               np.mat([[np.cos(self.theta) * forwardDist], [np.sin(self.theta) * forwardDist]]) + \
               np.mat([[np.sin(self.theta) * latDist], [-np.cos(self.theta) * latDist]])
        car2 = np.mat([[self.boundX[1]], [self.boundY[1]]]) + \
               np.mat([[np.cos(self.theta) * forwardDist], [np.sin(self.theta) * forwardDist]]) - \
               np.mat([[np.sin(self.theta) * latDist], [-np.cos(self.theta) * latDist]])
        car3 = np.mat([[self.boundX[3]], [self.boundY[3]]]) - \
               np.mat([[np.cos(self.theta) * longDist], [np.sin(self.theta) * longDist]]) + \
               np.mat([[np.sin(self.theta) * latDist], [-np.cos(self.theta) * latDist]])
        car4 = np.mat([[self.boundX[2]], [self.boundY[2]]]) - \
               np.mat([[np.cos(self.theta) * longDist], [np.sin(self.theta) * longDist]]) - \
               np.mat([[np.sin(self.theta) * latDist], [-np.cos(self.theta) * latDist]])
        x = [car1[0, 0], car2[0, 0], car4[0, 0], car3[0, 0], car1[0, 0]]
        y = [car1[1, 0], car2[1, 0], car4[1, 0], car3[1, 0], car1[1, 0]]'''
        return self.carBox()


    def cenPos(self):
        x0 = np.mat([[self.posx], [self.posy]])
        Rot0 = np.mat([[np.cos(self.theta), -np.sin(self.theta)],
                       [np.sin(self.theta), np.cos(self.theta)]])
        centerCar0 = x0 + Rot0 * np.mat([[self.offset], [0]])
        return centerCar0[0, 0], centerCar0[1, 0]


    def stateupdate(self, action):
        if not self.endFlag:
            v_temp = self.vel + action * T
            rand_s = np.random.normal(0, 2e-7 + self.vel * T / (3 * 4))

            if 0 <= v_temp <= V_max:
                self.vel += action * T
                s = self.vel * T + 1 / 2 * action * T ** 2
            elif v_temp > V_max:
                v_temp = V_max
                s = 0.5 * (self.vel + v_temp) * T
                self.vel = V_max
            else:
                v_temp = 0
                s = 0.5 * (self.vel + v_temp) * T
                self.vel = 0
            s = s + rand_s if s + rand_s > 0 else 0

            if self.trafficModel.flag == 'DR':
                self.DR(s)
            elif self.trafficModel.flag == 'DU':
                self.DU(s)
            elif self.trafficModel.flag == 'DL':
                self.DL(s)
            elif self.trafficModel.flag == 'RU':
                self.RU(s)
            elif self.trafficModel.flag == 'RL':
                self.RL(s)
            elif self.trafficModel.flag == 'RD':
                self.RD(s)
            elif self.trafficModel.flag == 'LD':
                self.LD(s)
            elif self.trafficModel.flag == 'LR':
                self.LR(s)
            elif self.trafficModel.flag == 'LU':
                self.LU(s)
            elif self.trafficModel.flag == 'UL':
                self.UL(s)
            elif self.trafficModel.flag == 'UD':
                self.UD(s)
            elif self.trafficModel.flag == 'UR':
                self.UR(s)

            self.cen_x, self.cen_y = self.cenPos()
            self.boundX, self.boundY = self.carBox()
            self.safeX, self.safeY = self.carSafeBox()
            self.endFlag = self.targetCheck()



    def getRelPos(self):
        if self.trafficModel.flag == 'DR':
            if self.posy < self.middlePoint[1]:
                pos = self.middlePoint[1] - self.posy
            else:
                pos = self.middlePoint[0] - self.posx

        elif self.trafficModel.flag == 'DU':
            if self.posy < self.middlePoint[1]:
                pos = self.middlePoint[1] - self.posy
            else:
                pos = self.middlePoint[1] - self.posy

        elif self.trafficModel.flag == 'DL':
            if self.posy < self.middlePoint[1]:
                pos = self.middlePoint[1] - self.posy
            else:
                pos = - (self.middlePoint[0] - self.posx)

        elif self.trafficModel.flag == 'RU':
            if self.posx > self.middlePoint[0]:
                pos = - (self.middlePoint[0] - self.posx)
            else:
                pos = self.middlePoint[1] - self.posy

        elif self.trafficModel.flag == 'RL':
            if self.posx > self.middlePoint[0]:
                pos = - (self.middlePoint[0] - self.posx)
            else:
                pos = - (self.middlePoint[0] - self.posx)

        elif self.trafficModel.flag == 'RD':
            if self.posx > self.middlePoint[0]:
                pos = - (self.middlePoint[0] - self.posx)
            else:
                pos = - (self.middlePoint[1] - self.posy)

        elif self.trafficModel.flag == 'LD':
            if self.posx < self.middlePoint[0]:
                pos = self.middlePoint[0] - self.posx
            else:
                pos = - (self.middlePoint[1] - self.posy)

        elif self.trafficModel.flag == 'LR':
            if self.posx < self.middlePoint[0]:
                pos = self.middlePoint[0] - self.posx
            else:
                pos = self.middlePoint[0] - self.posx

        elif self.trafficModel.flag == 'LU':
            if self.posx < self.middlePoint[0]:
                pos = self.middlePoint[0] - self.posx
            else:
                pos = self.middlePoint[1] - self.posy

        elif self.trafficModel.flag == 'UL':
            if self.posy > self.middlePoint[1]:
                pos = - (self.middlePoint[1] - self.posy)
            else:
                pos = - (self.middlePoint[0] - self.posx)

        elif self.trafficModel.flag == 'UD':
            if self.posy > self.middlePoint[1]:
                pos = - (self.middlePoint[1] - self.posy)
            else:
                pos = - (self.middlePoint[1] - self.posy)

        else:
            if self.posy > self.middlePoint[1]:
                pos = - (self.middlePoint[1] - self.posy)
            else:
                pos = self.middlePoint[0] - self.posx

        return pos


    def targetCheck(self):
        if abs(self.posy - self.ref[3][1]) < epsilon and abs(self.posx - self.ref[3][0]) < epsilon:
            return True
        else:
            return False


    def DR(self, s):

        if self.posy < self.ref[1][1] and self.posx == self.ref[1][0]:
            x=self.posx
            y=self.posy+s

            if y< self.ref[1][1]:
                self.posy = y
                self.posx = x
            if y > self.ref[1][1]:
                self.alpha = abs(y-self.ref[1][1])/self.R_max  #这里的alpha下面可以用吗
                self.posx = self.ref[2][0] - self.R_max * cos(self.alpha)
                self.posy = self.ref[1][1] + self.R_max * sin(self.alpha)
            self.delta = 0
            self.theta = pi / 2
            self.history.append((self.posx, self.posy, self.theta))

        if  self.posx >= self.ref[1][0] and self.ref[1][1] <= self.posy < self.ref[2][1]:
            self.alpha = self.alpha + s / self.R_max   #这里怎么调用上面的alpha
            x = self.ref[2][0] - self.R_max * cos(self.alpha)
            y = self.ref[1][1] + self.R_max * sin(self.alpha)
            if x < self.ref[2][0]:
                self.posy = y
                self.posx = x
            if x > self.ref[2][0]:
                beta = asin(abs( x - self.ref[2][0])/self.R_max)
                self.posy = self.ref[2][1]
                self.posx = self.ref[2][0]+ self.R_max*beta
            self.delta = delta_max
            self.theta = pi / 2 - self.alpha
            self.history.append((self.posx, self.posy, self.theta))
        if  self.posy == self.ref[2][1]:
            x = self.posx + s
            y = self.ref[2][1]
            if x < self.ref[3][0]:
                self.posy = y
                self.posx = x
            if x >= self.ref[3][0]:
                self.posy = self.ref[3][1]
                self.posx = self.ref[3][0]
            self.delta = 0
            self.theta = 0
            self.history.append((self.posx, self.posy, self.theta))


    def DU(self, s):

        if self.posy < self.ref[3][1] and self.posx == self.ref[1][0]:
            x = self.posx
            y = self.posy + s

            if y < self.ref[3][1]:
                self.posy = y
                self.posx = x
            if y >= self.ref[3][1]:
                self.posy = self.ref[3][1]
                self.posx = self.ref[3][0]
            self.delta = 0
            self.theta = pi / 2
            self.history.append((self.posx, self.posy, self.theta))


    def DL(self, s):
        if self.posy < self.ref[1][1] and self.posx == self.ref[1][0]:
            x = self.posx
            y = self.posy + s

            if y < self.ref[1][1]:
                self.posy = y
                self.posx = x
            if y >= self.ref[1][1]:
                alpha = abs(y - self.ref[1][1]) / self.R_max
                self.posx = self.ref[2][0] + self.R_max * cos(self.alpha) # it was minus in ‘DR’
                self.posy = self.ref[1][1] + self.R_max * sin(self.alpha)
            self.delta = 0
            self.theta = pi / 2
            self.history.append((self.posx, self.posy, self.theta))
        # self.delta = delta_max
        elif self.posx <= self.ref[1][0] and self.ref[1][1] <= self.posy < self.ref[2][1]:
            self.alpha = self.alpha + s / self.R_max
            x = self.ref[2][0] + self.R_max * cos(self.alpha)
            y = self.ref[1][1] + self.R_max * sin(self.alpha)
            if x > self.ref[2][0]:
                self.posy = y
                self.posx = x
            if x < self.ref[2][0]:
                beta = asin(abs(x - self.ref[2][0]) / self.R_max)
                self.posy = self.ref[2][1]
                self.posx = self.ref[2][0] - self.R_max * beta
            self.delta = delta_max
            self.theta = pi / 2 + self.alpha
            self.history.append((self.posx, self.posy, self.theta))

        elif self.posy == self.ref[2][1]:
            x = self.posx - s
            y = self.ref[2][1]
            if x > self.ref[3][0]:
                self.posy = y
                self.posx = x
            if x <= self.ref[3][0]:
                self.posy = self.ref[3][1]
                self.posx = self.ref[3][0]
            self.delta = 0
            self.theta = - pi
            self.history.append((self.posx, self.posy, self.theta))


    def RU(self, s):

        if self.posx > self.ref[1][0] and self.posy == self.ref[1][1]:
            x = self.posx - s
            y = self.posy

            if x > self.ref[1][0]:
                self.posy = y
                self.posx = x
            if x <= self.ref[1][0]:
                self.alpha = abs(x - self.ref[1][0]) / self.R_max  # 这里的alpha下面可以用吗
                self.posx = self.ref[1][0] - self.R_max * sin(self.alpha)
                self.posy = self.ref[2][1] - self.R_max * cos(self.alpha)
            self.delta = 0
            self.theta = - pi
            self.history.append((self.posx, self.posy, self.theta))

        if self.posx <= self.ref[1][0] and self.posy < self.ref[2][1]:
            self.alpha = self.alpha + s / self.R_max  # 这里怎么调用上面的alpha
            x = self.ref[1][0] - self.R_max * sin(self.alpha)
            y = self.ref[2][1] - self.R_max * cos(self.alpha)
            if y < self.ref[2][1]:
                self.posy = y
                self.posx = x
            if y > self.ref[2][1]:
                beta = asin(abs(y - self.ref[2][1]) / self.R_max)
                self.posy = self.ref[2][1] + self.R_max * beta
                self.posx = self.ref[2][0]
            self.delta = delta_max
            self.theta = pi  - self.alpha
            self.history.append((self.posx, self.posy, self.theta))
        if self.posx == self.ref[2][0]:
            y = self.posy + s
            x = self.ref[2][0]
            if y < self.ref[3][1]:
                self.posy = y
                self.posx = x
            if y >= self.ref[3][1]:
                self.posy = self.ref[3][1]
                self.posx = self.ref[3][0]
            self.delta = 0
            self.theta = pi / 2
            self.history.append((self.posx, self.posy, self.theta))


    def RL(self, s):

        if self.posx >= self.ref[3][0] and self.posy == self.ref[1][1]:
            x = self.posx - s
            y = self.posy

            if x > self.ref[3][0]:
                self.posy = y
                self.posx = x
            if  x <= self.ref[3][0]:
                self.posy = self.ref[3][1]
                self.posx = self.ref[3][0]
            self.delta = 0
           # self.theta = -pi
            self.history.append((self.posx, self.posy, self.theta))


    def RD(self, s):
        if self.posx > self.ref[1][0] and self.posy == self.ref[1][1]:
            x = self.posx - s
            y = self.posy
            if x > self.ref[1][0]:
                self.posy = y
                self.posx = x
            if x <= self.ref[1][0]:
                self.alpha = abs(x - self.ref[1][0]) / self.R_max  # 这里的alpha下面可以用吗
                self.posx = self.ref[1][0] - self.R_max * sin(self.alpha)
                self.posy = self.ref[2][1] + self.R_max * cos(self.alpha)
            self.delta = 0
            #self.theta = - pi
            self.history.append((self.posx, self.posy, self.theta))

        if  self.ref[2][0] < self.posx <= self.ref[1][0] and self.posy >= self.ref[2][1]:
            self.alpha = self.alpha + s / self.R_max  # 这里怎么调用上面的alpha
            x = self.ref[1][0] - self.R_max * sin(self.alpha)
            y = self.ref[2][1] + self.R_max * cos(self.alpha)
            if y > self.ref[2][1]:
                self.posy = y
                self.posx = x
            if y <= self.ref[2][1]:
                beta = asin(abs(y - self.ref[2][1]) / self.R_max)
                self.posy = self.ref[2][1] - self.R_max * beta
                self.posx = self.ref[2][0]
            self.delta = delta_max
            self.theta = - (pi - self.alpha)
            self.history.append((self.posx, self.posy, self.theta))
        if self.posx == self.ref[2][0]:
            y = self.posy - s
            x = self.ref[2][0]
            if y > self.ref[3][1]:
                self.posy = y
                self.posx = x
            if y <= self.ref[3][1]:
                self.posy = self.ref[3][1]
                self.posx = self.ref[3][0]
            self.delta = 0
            self.theta = - pi / 2
            self.history.append((self.posx, self.posy, self.theta))


    def LD(self, s):

        if self.posx < self.ref[1][0] and self.posy == self.ref[1][1]:
            x = self.posx + s
            y = self.posy

            if x < self.ref[1][0]:
                self.posy = y
                self.posx = x
            if x >= self.ref[1][0]:
                self.alpha = abs(x - self.ref[1][0]) / self.R_max  # 这里的alpha下面可以用吗
                self.posx = self.ref[1][0] + self.R_max * sin(self.alpha)
                self.posy = self.ref[2][1] + self.R_max * cos(self.alpha)
            self.delta = 0
            self.theta = 0
            self.history.append((self.posx, self.posy, self.theta))

        if self.posx > self.ref[1][0] and self.posy > self.ref[2][1]:
            self.alpha = self.alpha + s / self.R_max  # 这里怎么调用上面的alpha
            x = self.ref[1][0] + self.R_max * sin(self.alpha)
            y = self.ref[2][1] + self.R_max * cos(self.alpha)
            if y > self.ref[2][1]:
                self.posy = y
                self.posx = x
            if y < self.ref[2][1]:
                beta = asin(abs(y - self.ref[2][1]) / self.R_max)
                self.posy = self.ref[2][1] - self.R_max * beta
                self.posx = self.ref[2][0]
            self.delta = delta_max
            self.theta = - self.alpha
            self.history.append((self.posx, self.posy, self.theta))
        if self.posx == self.ref[2][0]:
            y = self.posy - s
            x = self.ref[2][0]
            if y > self.ref[3][1]:
                self.posy = y
                self.posx = x
            if y <= self.ref[3][1]:
                self.posy = self.ref[3][1]
                self.posx = self.ref[3][0]
            self.delta = 0
            self.theta = -pi / 2
            self.history.append((self.posx, self.posy, self.theta))


    def LR(self, s):

        if self.posx < self.ref[3][0] and self.posy == self.ref[1][1]:
            x = self.posx + s
            y = self.posy

            if x < self.ref[3][0]:
                self.posy = y
                self.posx = x
            if x >= self.ref[3][0]:
                self.posy = self.ref[3][1]
                self.posx = self.ref[3][0]
            self.delta = 0
            self.theta = 0
            self.history.append((self.posx, self.posy, self.theta))


    def LU(self, s):

        if self.posx < self.ref[1][0] and self.posy == self.ref[1][1]:
            x = self.posx + s
            y = self.posy

            if x < self.ref[1][0]:
                self.posy = y
                self.posx = x
            if x >= self.ref[1][0]:
                self.alpha = abs(x - self.ref[1][0]) / self.R_max  # 这里的alpha下面可以用吗
                self.posx = self.ref[1][0] + self.R_max * sin(self.alpha)
                self.posy = self.ref[2][1] - self.R_max * cos(self.alpha)
            self.delta = 0
            self.theta = 0
            self.history.append((self.posx, self.posy, self.theta))

        if self.posx >= self.ref[1][0] and self.posy < self.ref[2][1]:
            self.alpha = self.alpha + s / self.R_max  # 这里怎么调用上面的alpha
            x = self.ref[1][0] + self.R_max * sin(self.alpha)
            y = self.ref[2][1] - self.R_max * cos(self.alpha)
            if y <self.ref[2][1]:
                self.posy = y
                self.posx = x
            if y >= self.ref[2][1]:
                beta = asin(abs(y - self.ref[2][1]) / self.R_max)
                self.posy = self.ref[2][1] + self.R_max * beta
                self.posx = self.ref[2][0]
            self.delta = delta_max
            self.theta =  self.alpha
            self.history.append((self.posx, self.posy, self.theta))
        if self.posx == self.ref[2][0]:
            y = self.posy + s
            x = self.ref[2][0]
            if y < self.ref[3][1]:
                self.posy = y
                self.posx = x
            if y >= self.ref[3][1]:
                self.posy = self.ref[3][1]
                self.posx = self.ref[3][0]
            self.delta = 0
            self.theta = pi / 2
            self.history.append((self.posx, self.posy, self.theta))


    def UL(self, s):

        if self.posy > self.ref[1][1]and self.posx == self.ref[1][0]:
            x=self.posx
            y=self.posy - s

            if y > self.ref[1][1]:
                self.posy = y
                self.posx = x
            if y < self.ref[1][1]:
                self.alpha = abs(y-self.ref[1][1])/self.R_max  #这里的alpha下面可以用吗
                self.posx = self.ref[2][0] + self.R_max * cos(self.alpha)
                self.posy = self.ref[1][1] - self.R_max * sin(self.alpha)
            self.delta = 0
            self.theta = -pi / 2
            self.history.append((self.posx, self.posy, self.theta))

        if  self.posx < self.ref[1][0] and self.posy > self.ref[2][1]:
            self.alpha = self.alpha + s / self.R_max   #这里怎么调用上面的alpha
            x = self.ref[2][0] + self.R_max * cos(self.alpha)
            y = self.ref[1][1] - self.R_max * sin(self.alpha)
            if x > self.ref[2][0]:
                self.posy = y
                self.posx = x
            if x < self.ref[2][0]:
                beta = asin(abs( x - self.ref[2][0])/self.R_max)
                self.posy = self.ref[2][1]
                self.theta = -pi / 2 - self.alpha
                self.posx = self.ref[2][0] - self.R_max*beta
            self.delta = delta_max
            self.theta = -pi / 2 - self.alpha
            self.history.append((self.posx, self.posy, self.theta))
        if  self.posy == self.ref[2][1]:
            x = self.posx - s
            y = self.ref[2][1]
            if x > self.ref[3][0]:
                self.posy = y
                self.posx = x
            if x <= self.ref[3][0]:
                self.posy = self.ref[3][1]
                self.posx = self.ref[3][0]
            self.delta = 0
            self.theta = -pi
            self.history.append((self.posx, self.posy, self.theta))


    def UD(self, s):

        if self.posy > self.ref[3][1] and self.posx == self.ref[1][0]:
            x = self.posx
            y = self.posy - s

            if y > self.ref[3][1]:
                self.posy = y
                self.posx = x

            if y <= self.ref[3][1]:
                self.posy = self.ref[3][1]
                self.posx = self.ref[3][0]
            self.delta = 0
            self.theta = -pi / 2
            self.history.append((self.posx, self.posy, self.theta))


    def UR(self, s):

        if self.posy > self.ref[1][1]and self.posx == self.ref[1][0]:
            x=self.posx
            y=self.posy - s

            if y > self.ref[1][1]:
                self.posy = y
                self.posx = x
            if y < self.ref[1][1]:
                self.alpha = abs(y-self.ref[1][1])/self.R_max  #这里的alpha下面可以用吗
                self.posx = self.ref[2][0] - self.R_max * cos(self.alpha)
                self.posy = self.ref[1][1] - self.R_max * sin(self.alpha)
            self.delta = 0
            self.theta = - pi/2
            self.history.append((self.posx, self.posy, self.theta))

        if  self.posx > self.ref[1][0] and self.posy > self.ref[2][1]:
            self.alpha = self.alpha + s / self.R_max   #这里怎么调用上面的alpha
            x = self.ref[2][0] - self.R_max * cos(self.alpha)
            y = self.ref[1][1] - self.R_max * sin(self.alpha)
            if x < self.ref[2][0]:
                self.posy = y
                self.posx = x
            if x >= self.ref[2][0]:
                beta = asin(abs( x - self.ref[2][0])/self.R_max)
                self.posy = self.ref[2][1]
                self.posx = self.ref[2][0] + self.R_max*beta
            self.delta = delta_max
            self.theta = -pi / 2 + self.alpha
            self.history.append((self.posx, self.posy, self.theta))
        if  self.posy == self.ref[2][1]:
            x = self.posx + s
            y = self.ref[2][1]
            if x < self.ref[3][0]:
                self.posy = y
                self.posx = x
            if x >= self.ref[3][0]:
                self.posy = self.ref[3][1]
                self.posx = self.ref[3][0]
            self.delta = 0
            self.theta = 0
            self.history.append((self.posx, self.posy, self.theta))
