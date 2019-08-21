#!/bin/env python
# -*- coding: utf-8 -*-
import os

import numpy as np
import numpy.ma as ma

from recsys.common.cache import cached


class TestCache(object):

    def test_array_cached(self, tmpdir):
        def _test_do(arg):
            if arg == 1:
                return [1,2,3]
            else:
                return [2,3,4]
        cache_file = os.path.join(tmpdir, "test.npy")
        cache_func = cached(cache_file)(_test_do)
        cache_func(1)

        assert os.path.exists(cache_file)
        result = cache_func(2)
        assert np.all(result == [1,2,3])

        result = cache_func(3)

    def test_ma_cached(self, tmpdir):
        def _test_do(arg):
            if arg == 1:
                return ma.masked_equal([0,1,2], 0)
            else:
                return ma.masked_equal([1,2,3], 0)
        cache_file = os.path.join(tmpdir, "test.npy")
        cache_func = cached(cache_file)(_test_do)
        result = cache_func(1)

        assert os.path.exists(cache_file)
        result2 = cache_func(2)
        assert np.all(result == result2)

        result3 = cache_func(3)

    def test_np_cached(self, tmpdir):
        def _test_do(arg):
            if arg == 1:
                return np.array([0,1,2])
            else:
                return np.array([1,2,3])
        cache_file = os.path.join(tmpdir, "test.npy")
        cache_func = cached(cache_file)(_test_do)
        result = cache_func(1)

        assert os.path.exists(cache_file)
        result2 = cache_func(2)
        assert np.all(result == result2)

        result3 = cache_func(3)
