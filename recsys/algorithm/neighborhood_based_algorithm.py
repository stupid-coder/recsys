#!/bin/env python
# -*- coding: utf-8
import time
from typing import NamedTuple

import numpy as np
import numpy.ma as ma

from recsys.algorithm.algorithm import Algorithm
from recsys.algorithm.predictor import PredictorFactory
from recsys.algorithm.similarity import SimilaritorFactory
from recsys.common.cache import cached

#SimilarityConfig = namedtuple("SimilarityConfig", ["name", "discounted_beta"])
# PredictorConfig = namedtuple("PredictorConfig", ["name"])
# NeighborhoodBasedConfig = namedtuple("NeighborhoodBasedConfig", ["sim_config", "topk", "sim_threshold", "predictor_config"])

class SimilarityConfig(NamedTuple):
    name: str
    discounted_beta: int = None
    amplify_alpha: float = None

class PredictorConfig(NamedTuple):
    name: str

class NeighborhoodBasedConfig(NamedTuple):
    sim_config: SimilarityConfig
    predictor_config: PredictorConfig
    topk: int = None
    sim_threshold: float = None


class NeighborhoodBasedAlgorithm(Algorithm):
    def __init__(self, name, config):
        super().__init__(name, config)

    def __fit__(self, rating):
        print("__fit__ start")
        # rating 数据集
        if isinstance(rating, ma.MaskedArray):
            self._rating = rating
        else:
            self._rating = ma.masked_equal(rating, 0)
        self._mean = ma.mean(self._rating, axis=1, keepdims=True)
        self._sigma = ma.std(self._rating, axis=1, keepdims=True)
        self._mean_center_rating = self._rating - self._mean
        self._z = self.mean_center_rating / self._sigma
        similaritor = SimilaritorFactory(self.name, self.config)
        self._sim = similaritor(self.rating)
        print("__fit__ end")

    def __predict__(self):
        print("__predict__ start")
        rating_hat = np.zeros(self._rating.shape, dtype=np.float)

        start = time.clock()
        rows, cols = self.rating.shape

        predictor = PredictorFactory(self.config.predictor_config)

        for i in range(rows):
            if i % 10 == 0:
                print("[__predict__:{:.2f}s] {},{} {}%".format((time.clock()-start),i, rows, i * 100 / rows))
            if self.config.topk is not None:
                neighborhood = [el for el in np.argsort(self.sim[i]) if el != i][:-self.config.topk-1:-1]
            elif self.config.sim_threshold is not None:
                neighborhood = [el for el in np.argsort(self.sim[i]) if el != i and self.sim[i][el] > self.config.sim_threshold]
            else:
                raise RuntimeError("topk or sim_threshold must be setted")

            if len(neighborhood) == 0:
                continue

            rating_hat[i] = predictor(rating=self.rating[neighborhood],
                                      mean=self.mean[i],
                                      sigma=self.sigma[i],
                                      mean_center_rating=self.mean_center_rating[neighborhood],
                                      z=self.z[neighborhood],
                                      sim=self.sim[i,neighborhood][:, np.newaxis])
        print("__predict__ end")
        return rating_hat


class UserBasedAlgorithm(NeighborhoodBasedAlgorithm):

    def __init__(self, config):
        super().__init__("UserBasedAlgorithm", config)

    def fit(self, rating):
        self.__fit__(rating)

    def predict(self, data=None):
        return self.__predict__()


class ItemBasedAlgorithm(NeighborhoodBasedAlgorithm):

    def __init__(self, config):
        super().__init__("ItemBasedAlgorithm", config)

    def fit(self, rating):
        self.__fit__(rating.T)

    def predict(self, data=None):
        return self.__predict__().T
