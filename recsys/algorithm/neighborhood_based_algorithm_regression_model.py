#!/bin/env python
# -*- coding: utf-8 -*-


import logging
import time
from typing import NamedTuple

import numpy as np
from numpy import ma

from recsys.algorithm.algorithm import Algorithm
from recsys.algorithm.similarity import person

logger = logging.getLogger(__name__)

class RegressionModelNeighborhoodBasedConfig(NamedTuple):
    topk: int = None
    lr: float = 0.001
    epochs: int = 1000
    wdecay: float = 0
    check_gradient: bool = False

class RegressionModelNeighborhoodBasedAlgorithm(Algorithm):
    def __init__(self, name, config):
        super().__init__(name, config)

    def __forward__(self, j): #,ngb_rating, ngb_weight, m_bias, ngb_m_bias, ngb_n_bias):
        _rating = self._rating[:, j]
        _ngb_rating = _rating[self._neighborhood.flat].reshape((self._m, -1))
        _ngb_m_bias = self._m_bias[self._neighborhood.flat].reshape((self._m, -1))
        _ngb_n_bias = self._n_bias[j]

        _adjust_rating = _ngb_rating - _ngb_m_bias - _ngb_n_bias
        _adjust_factor = np.sqrt(ma.count(_ngb_rating, axis=1))
        _hat_rating = self._m_bias + _ngb_n_bias + ma.sum(self._weight * _adjust_rating, axis=1) / _adjust_factor
        return _hat_rating, (_adjust_rating, _adjust_factor, _ngb_rating, _ngb_m_bias, _ngb_n_bias)

    def __backward__(self, hat_rating, rating, adjust_rating, adjust_factor):
        _ngb_weight = self._weight * ma.logical_not(ma.getmaskarray(adjust_rating)).astype(np.int)
        _g_l = hat_rating - rating
        _adjust_g_l = _g_l / adjust_factor
        _adjust_g_l = _adjust_g_l[:, np.newaxis]
        _g_m_bias = _g_l
        _g_ngb_m_bias = -1.0 * _ngb_weight * _adjust_g_l
        _g_ngb_n_bias = np.sum(_g_l * (1 - np.sum(_ngb_weight, axis=1) / adjust_factor))
        _g_weight = adjust_rating * _adjust_g_l
        return _g_m_bias, _g_ngb_m_bias, _g_ngb_n_bias, _g_weight

    def __check_gradient__(self, j, g_weight, g_m_bias, g_ngb_n_bias):

        it = np.nditer(g_weight, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            if g_weight[idx] is ma.masked:
                it.iternext()
                continue
            self._weight[idx] += 1e-6
            hat_rating1, _ = self.__forward__(j)
            self._weight[idx] -= 2e-6
            hat_rating2, _ = self.__forward__(j)
            _g_ngb_weight = (0.5 * ma.sum(ma.power(self._rating[:, j] - hat_rating1, 2)) - 0.5 * ma.sum(ma.power(self._rating[:, j] - hat_rating2, 2))) / 2e-6
            self._weight[idx] += 1e-6
            assert np.all(np.isclose(_g_ngb_weight, g_weight[idx], rtol=1e-2))
            it.iternext()

        for i in range(self._m_bias.size):
            if g_m_bias[i] is ma.masked:
                continue
            self._m_bias[i] += 1e-6
            hat_rating1, _ = self.__forward__(j)
            self._m_bias[i] -= 2e-6
            hat_rating2, _ = self.__forward__(j)
            _g_m_bias = (0.5 * ma.sum(ma.power(self._rating[:, j] - hat_rating1, 2)) - 0.5 * ma.sum(ma.power(self._rating[:, j] - hat_rating2, 2))) / 2e-6
            self._m_bias[i] += 1e-6
            assert np.all(np.isclose(_g_m_bias, g_m_bias[i], rtol=1e-2))

        self._n_bias[j] += 1e-6
        hat_rating1, _ = self.__forward__(j)
        self._n_bias[j] -= 2e-6
        hat_rating2, _ = self.__forward__(j)
        _g_ngb_n_bias = (0.5 * ma.sum(ma.power(self._rating[:, j] - hat_rating1, 2)) - 0.5 * ma.sum(ma.power(self._rating[:, j] - hat_rating2, 2))) / 2e-6
        self._n_bias[j] += 1e-6
        assert np.all(np.isclose(_g_ngb_n_bias, g_ngb_n_bias, rtol=1e-2))

    def __fit__(self, rating, row=True):
        if isinstance(rating, ma.MaskedArray):
            self._rating = rating
        else:
            self._rating = ma.masked_equal(rating, 0)

        self._mean = ma.mean(self._rating, axis=1, keepdims=True)
        self._mean_center_rating = self._rating - self._mean
        self._rating_filled = self._rating.filled(0)

        if row:
            self._sim = person(mean_center_rating=self._mean_center_rating)
        else:
            self._rating = self._rating.T
            self._mean_center_rating = self._mean_center_rating.T
            self._sim = person(mean_center_rating=self._mean_center_rating)

        self._sim[np.diag_indices(self._sim.shape[0])] = -999

        # params
        self._neighborhood = np.argsort(self._sim, axis=1)[:, -self.config.topk:]
        self._neighborhood_idx = ([int(i/self._neighborhood.shape[1]) for i in range(self._neighborhood.size)], self._neighborhood.flatten())

        if row:
            self._m, self._n = rating.shape
        else:
            self._n, self._m = rating.shape

        self._weight = np.random.randn(self._m, self.config.topk)
        self._m_bias = np.random.randn(self._m)
        self._n_bias = np.random.randn(self._n)

        for epoch in range(self.config.epochs):
            _epoch_loss = 0.0
            start = time.perf_counter()

            for j in range(self._n):
                # forward
                _hat_rating, mid_data = self.__forward__(j)
                _loss = 0.5 * ma.sum(ma.power(self._rating[:, j]  - _hat_rating, 2))
                _epoch_loss += _loss

                # backward
                _g_m_bias, _g_ngb_m_bias, _g_ngb_n_bias, _g_weight = self.__backward__(_hat_rating, self._rating[:, j], mid_data[0], mid_data[1])

                for i, g in zip(self._neighborhood.flat, _g_ngb_m_bias.flat):
                    if g is not ma.masked:
                        _g_m_bias[i] += g

                # check gradient
                self.__check_gradient__(j, _g_weight, _g_m_bias, _g_ngb_n_bias)

                # update gradient
                self._m_bias -= self.config.lr * _g_m_bias + self.config.wdecay * self._m_bias
                self._n_bias[j] -= self.config.lr * _g_ngb_n_bias + self.config.wdecay * self._n_bias[j]
                self._weight -= self.config.lr * _g_weight + self.config.wdecay * self._weight

            _epoch_loss /= ma.count(self._rating)
            logger.debug("[{:4d} epoch\t{:.2f}s] loss:{:.2f}".format(epoch, time.perf_counter()-start, _epoch_loss))


    def __predict__(self):
        hat_rating = ma.masked_all((self._m, self._n))
        for j in range(self._n):
            hat_rating[:, j], _ = self.__forward__(j)
        return hat_rating


    def __parameters__(self):
        return {
            "m_bias": self._m_bias,
            "n_bias": self._n_bias,
            "weight": self._weight
        }


class UserBasedRegressionModel(RegressionModelNeighborhoodBasedAlgorithm):
    def __init__(self, config):
        super().__init__("UserBasedRegressionModel", config)

    def fit(self, rating):
        self.__fit__(rating)

    def predict(self, rating=None):
        return self.__predict__()


class ItemBasedRegressionModel(RegressionModelNeighborhoodBasedAlgorithm):
    def __init__(self, config):
        super().__init__("ItemBasedRegressionModel", config)

    def fit(self, rating):
        self.__fit__(rating, row=False)

    def predict(self, rating=None):
        return self.__predict__().T
