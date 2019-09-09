#!/bin/env python
# -*- coding: utf-8

import numpy as np
import numpy.ma as ma


def statistic(y, y_hat):
    src_count, hat_count = y.count(), y_hat.count()
    common_count = np.sum(ma.logical_or(ma.getmaskarray(y), ma.getmaskarray(y_hat)).astype(np.int))
    return src_count, hat_count, common_count/src_count

def rmse(y, y_hat):
    return ma.sqrt(ma.mean(ma.power(y-y_hat, 2)))
