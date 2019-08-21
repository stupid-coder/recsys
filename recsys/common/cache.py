#!/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import numpy.ma as ma


def cached(cache_path):
    def cache_wrapper(func):
        def _do(*argc, **argv):
            print(_do.__dict__)
            if cache_path in _do.__dict__:
                print("DEBUG: return")
                return _do.__dict__[cache_path]
            else:
                if os.path.exists(cache_path):
                    print("DEBUG: load")
                    result = np.load(open(cache_path, "rb"), allow_pickle=True)
                    _do.__dict__[cache_path] = result
                else:
                    print("DEBUG: func")
                    _do.__dict__[cache_path] = func(*argc, **argv)
                    if isinstance(_do.__dict__[cache_path], (ma.MaskedArray, np.ndarray)):
                        _do.__dict__[cache_path].dump(cache_path)
                    else:
                        np.save(open(cache_path, "wb"), _do.__dict__[cache_path])
            return _do.__dict__[cache_path]
        return _do
    return cache_wrapper
