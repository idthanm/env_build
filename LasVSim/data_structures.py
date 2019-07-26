# coding=utf-8
from __future__ import print_function
from ctypes import *
import os
# from _ctypes import FreeLibrary

DEFAULT_SETTING_FILE = 'Library/simulation_setting_file.xml'
"""Simulation Setting Value"""

current_path = os.path.dirname(__file__)

TRAFFIC_TYPE = ['No Traffic', 'Mixed Traffic', 'Vehicle Only Traffic'] # 不能更改顺序
NO_TRAFFIC = 0
MIXED_TRAFFIC = 1
VEHICLE_ONLY_TRAFFIC = 2
TRAFFIC_DENSITY = ['Sparse', 'Middle', 'Dense'] # 不能更改顺序
SPARSE = 0
MIDDLE = 1
DENSE = 2

FILE_TYPE = ['C/C++ DLL', 'Python Module'] # 不能更改顺序

MAPS = ['Map1_Urban Road', 'Map2_Highway', 'Map3_Shanghai Anting',
        'Map4_Beijing Changping', 'Map5_Mcity', 'Map3_Highway_v2']  # 不能更改顺序
