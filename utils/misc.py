#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/8/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: misc.py
# =====================================

import os
import random
import subprocess
import time

import numpy as np


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def random_choice_with_index(obj_list):
    obj_len = len(obj_list)
    random_index = random.choice(list(range(obj_len)))
    random_value = obj_list[random_index]
    return random_value, random_index


def judge_is_nan(list_of_np_or_tensor):
    for m in list_of_np_or_tensor:
        if hasattr(m, 'numpy'):
            if np.any(np.isnan(m.numpy())):
                print(list_of_np_or_tensor)
                raise ValueError
        else:
            if np.any(np.isnan(m)):
                print(list_of_np_or_tensor)
                raise ValueError


class TimerStat:
    def __init__(self, window_size=10):
        self._window_size = window_size
        self._samples = []
        self._units_processed = []
        self._start_time = None
        self._total_time = 0.0
        self.count = 0

    def __enter__(self):
        assert self._start_time is None, "concurrent updates not supported"
        self._start_time = time.time()

    def __exit__(self, type, value, tb):
        assert self._start_time is not None
        time_delta = time.time() - self._start_time
        self.push(time_delta)
        self._start_time = None

    def push(self, time_delta):
        self._samples.append(time_delta)
        if len(self._samples) > self._window_size:
            self._samples.pop(0)
        self.count += 1
        self._total_time += time_delta

    def push_units_processed(self, n):
        self._units_processed.append(n)
        if len(self._units_processed) > self._window_size:
            self._units_processed.pop(0)

    def has_units_processed(self):
        return len(self._units_processed) > 0

    @property
    def mean(self):
        if not self._samples:
            return 0.0
        return float(np.mean(self._samples))

    @property
    def mean_units_processed(self):
        if not self._units_processed:
            return 0.0
        return float(np.mean(self._units_processed))

    @property
    def mean_throughput(self):
        time_total = float(sum(self._samples))
        if not time_total:
            return 0.0
        return float(sum(self._units_processed)) / time_total


def image2video(forder):
    os.chdir(forder)
    subprocess.call(['ffmpeg', '-framerate', '10', '-i', 'step%03d.png', 'video.mp4'])


