#!/bin/env python
# -*- coding: utf-8 -*-

from recsys.dataset.dataset import MovieLenDataset


class TestDataset(object):
    def test_R(self):
        ds = MovieLenDataset("data/ml-1m")
        r = ds.R
        print(type(r))
