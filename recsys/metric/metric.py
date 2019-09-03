#!/bin/env python
# -*- coding: utf-8

import numpy as np
import numpy.ma as ma


def rmse(y, y_hat):
    return ma.sqrt(ma.mean(ma.power(y-y_hat, 2)))
