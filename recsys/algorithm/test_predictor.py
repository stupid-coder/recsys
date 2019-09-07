#!/bin/env python
# -*- coding: utf-8 -*-
import math

import numpy as np
import numpy.ma as ma

from recsys.algorithm.predictor import mean_center_predictor, norm_predictor


def test_norm_predictor():
    rating = np.array([7, 6])
    sim = np.array([0.894, 0.939])
    assert math.isclose(norm_predictor(rating, sim), 6.49, rel_tol=1e-2)

    rating = np.array([4, 4])
    sim = np.array([0.894, 0.939])
    assert math.isclose(norm_predictor(rating, sim), 4)

def test_mean_center_predictor():
    assert math.isclose(mean_center_predictor(2, np.array([1.5, 1.2]), np.array([0.894, 0.939])), 3.35, rel_tol=1e-2)

    assert math.isclose(mean_center_predictor(2, np.array([-1.5, -0.8]), np.array([0.894, 0.939])), 0.86, rel_tol=1e-2)
