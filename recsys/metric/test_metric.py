#/bin/env python
# -*- coding: utf-8 -*-

import math

import numpy as np
import numpy.ma as ma

from recsys.metric import metric


class TestMetric(object):

    def test_rmse(self):
        y = ma.masked_equal([1,2,0,1], 0)
        y_hat = ma.masked_equal([0,1,2,2], 0)
        assert math.isclose(metric.rmse(y, y_hat), 1)
