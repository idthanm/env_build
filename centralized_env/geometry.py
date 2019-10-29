import numpy as np


def determinant(x1, x2, y1, y2):
    return x1 * y2 - x2 * y1


def intersect(l1, l2):
    aa = l1.point1; bb = l1.point2; cc = l2.point1; dd = l2.point2
    delta = determinant(bb[0] - aa[0], cc[0] - dd[0], bb[1] - aa[1], cc[1] - dd[1])
    if delta >= - (1e-6) and delta <= (1e-6):
        return False

    namenda = determinant(cc[0] - aa[0], cc[0] - dd[0], cc[1] - aa[1], cc[1] - dd[1]) / delta
    if namenda > 1 or namenda < 0:
        return False

    miu = determinant(bb[0] - aa[0], cc[0] - aa[0], bb[1] - aa[1], cc[1] - aa[1]) / delta
    if miu > 1 or miu < 0:
        return False
    return True


def intersect2(l1, l2):
    aa = l1.point1; bb = l1.point2; cc = l2.point1; dd = l2.point2
    ACX = cc[0] - aa[0]; ACY = cc[1] - aa[1]
    ABX = bb[0] - aa[0]; ABY = bb[1] - aa[1]
    ADX = dd[0] - aa[0]; ADY = dd[1] - aa[1]
    flag1 = (determinant(ACX, ABX, ACY, ABY) * determinant(ADX, ABX, ADY, ABY) < - (1e-6))
    CAX = aa[0] - cc[0]; CAY = aa[1] - cc[1]
    CDX = dd[0] - cc[0]; CDY = dd[1] - cc[1]
    CBX = bb[0] - cc[0]; CBY = bb[1] - cc[1]
    flag2 = (determinant(CAX, CDX, CAY, CDY) * determinant(CBX, CDX, CBY, CDY) < - (1e-6))
    if flag1 and flag2:
        return True
    else:
        return False



def rectCheck(X1, Y1, X2, Y2):
    rect1 = rect(X1, Y1)
    rect2 = rect(X2, Y2)
    for cl1 in rect1.crossLine:
        for bl2 in rect2.boundLine:
            if intersect2(cl1, bl2):
                return True
    for cl2 in rect2.crossLine:
        for bl1 in rect1.boundLine:
            if intersect2(cl2, bl1):
                return True
    return False


class rect(object):
    def __init__(self, X1, Y1):
        self.crossLine = [line(X1[0], Y1[0], X1[2], Y1[2]), line(X1[1], Y1[1], X1[3], Y1[3])]
        self.boundLine = [line(X1[0], Y1[0], X1[1], Y1[1]),
                          line(X1[1], Y1[1], X1[2], Y1[2]),
                          line(X1[2], Y1[2], X1[3], Y1[3]),
                          line(X1[3], Y1[3], X1[0], Y1[0])]


class line(object):
    def __init__(self, x1, y1, x2, y2):
        self.point1 = [x1, y1]
        self.point2 = [x2, y2]


def carBox(x0, phi, w, l):
    car1 = x0[0:2] + np.mat([[np.cos(phi) * l], [np.sin(phi) * l]]) + np.mat([[np.sin(phi) * w], [-np.cos(phi) * w]])
    car2 = x0[0:2] + np.mat([[np.cos(phi) * l], [np.sin(phi) * l]]) - np.mat([[np.sin(phi) * w], [-np.cos(phi) * w]])
    car3 = x0[0:2] - np.mat([[np.cos(phi) * l], [np.sin(phi) * l]]) + np.mat([[np.sin(phi) * w], [-np.cos(phi) * w]])
    car4 = x0[0:2] - np.mat([[np.cos(phi) * l], [np.sin(phi) * l]]) - np.mat([[np.sin(phi) * w], [-np.cos(phi) * w]])
    x = [car1[0,0],car2[0,0],car4[0,0],car3[0,0],car1[0,0]]
    y = [car1[1,0],car2[1,0],car4[1,0],car3[1,0],car1[1,0]]
    return x, y