#!/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np


def cached(cache_path):
    def cache_wrapper(func):
        def _do(*argc, **argv):
            print("[DEBUG] {} {}".format(argc, argv))
            if os.path.exists(cache_path):
                return np.load(open(cache_path, "rb"))
            else:
                result = func(*argc, **argv)
                np.save(open(cache_path, "wb"), result)
                return result
        return _do
    return cache_wrapper
