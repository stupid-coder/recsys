#!/bin/env python
# -*- coding: utf-8 -*-
import os

import numpy as np

from recsys.common.cache import cached


class TestCache(object):

    def test_cached(self, tmpdir):
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
