#!/bin/env python
# -*- coding: utf-8 -*-

import logging
import os

import numpy as np
import numpy.ma as ma

logger = logging.getLogger(__name__)

def cached(cache_path):
    def cache_wrapper(func):
        def _do(*argc, **argv):
            if cache_path in _do.__dict__:
                return _do.__dict__[cache_path]
            else:
                if os.path.exists(cache_path):
                    result = np.load(open(cache_path, "rb"), allow_pickle=True)
                    _do.__dict__[cache_path] = result
                else:
                    _do.__dict__[cache_path] = func(*argc, **argv)
                    if isinstance(_do.__dict__[cache_path], (ma.MaskedArray, np.ndarray)):
                        _do.__dict__[cache_path].dump(cache_path)
                    else:
                        np.save(open(cache_path, "wb"), _do.__dict__[cache_path])
            return _do.__dict__[cache_path]
        return _do
    return cache_wrapper
