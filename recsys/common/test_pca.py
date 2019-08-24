#!/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.decomposition import PCA

from recsys.common.pca import pca, pca2


class TestPCA(object):
    def test_pca(self):
        X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        print(pca(X, 1))
        print("--------------------------------------------------")
        print(pca2(X,1))
        # [[ 0.50917706]
        #  [ 2.40151069]
        #  [ 3.7751606 ]
        #  [-1.20075534]
        #  [-2.05572155]
        #  [-3.42937146]]
