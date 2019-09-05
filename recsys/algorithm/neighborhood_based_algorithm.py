#!/bin/env python
# -*- coding: utf-8
import time
from typing import NamedTuple

import numpy as np
import numpy.ma as ma

from recsys.algorithm.algorithm import Algorithm
from recsys.algorithm.predictor import PredictorFactory
from recsys.algorithm.similarity import SimilaritorFactory


class SimilarityConfig(NamedTuple):
    name: str
    discounted_beta: int = None
    amplify_alpha: float = None
    dims: int = None


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

    def __neighborhood__(self):
        raise NotImplementedError("[{}] __neighborhood__ not implemented".format(self.__class__))

    def __predict__(self):
        raise NotImplementedError("[{}] __predict__ not implemented".format(self.__class__))

    def fit(self, rating):
        if isinstance(rating, ma.MaskedArray):
            self._rating = rating
        else:
            self._rating = ma.masked_equal(rating, 0)
        self.__neighborhood__()

    def predict(self, rating=None):
        return self.__predict__()

class UserBasedAlgorithm(NeighborhoodBasedAlgorithm):

    def __init__(self, config):
        super().__init__("UserBasedAlgorithm", config)

    def __neighborhood__(self):
        self._mean = ma.mean(self._rating, axis=1, keepdims=True)
        self._sigma = ma.std(self._rating, axis=1, keepdims=True)
        self._mean_center_rating = self._rating - self._mean
        self._z = self._mean_center_rating / self._sigma

        assert self.config.sim_config.name in ["person", "discounted_person", "amplify_person", "idf_person", "pca_person"]

        similaritor = SimilaritorFactory(self.name, self.config)
        self._sim = similaritor(self._mean_center_rating.filled(0))
        sorted_neighborhood = ma.argsort(self._sim, axis=1, endwith=False)
        users_num, items_num = self._rating.shape

        self._neighborhood = []
        for i in range(users_num):
            for j in range(items_num):
                neighborhood = []
                if self.config.topk is not None:
                    neighborhood.append([el for el in sorted_neighborhood[i] if el != i and self._sim[i][el] is not ma.masked and self._rating[el][j] is not ma.masked][:-self.config.topk-1:-1])
                elif self.config.sim_threshold is not None:
                    neighborhood.append([el for el in sorted_neighborhood[i] if el != i and self._sim[i][el] > self.config.sim_threshold and self._rating[el][j] is not ma.masked])
                else:
                    raise RuntimeError("topk or sim_threshold must be setted")
            self._neighborhood.append(neighborhood)

    def __predict__(self):
        rating_hat = ma.masked_equal(np.zeros(self._rating.shape), 0)
        users_num, items_num = self._rating.shape
        predictor = PredictorFactory(self.config.predictor_config)

        start = time.clock()
        for i in range(users_num):
            if i % 10 == 0:
                print("[__predict__:{:.2f}s] {},{} {}%".format((time.clock()-start),i, users_num, i * 100 / users_num))

            for j in range(items_num):
                if self._neighborhood[i][j]:
                    continue

                rating_hat[i, j] = predictor(rating=self._rating[self._neighborhood[i][j], j],
                                             mean=self._mean[i],
                                             sigma=self._sigma[i],
                                             mean_center_rating=self._mean_center_rating[self._neighborhood[i][j], j],
                                             z=self._z[self._neighborhood[i][j], j],
                                             sim=self._sim[i, self._neighborhood[i][j]])
        return rating_hat


class ItemBasedAlgorithm(NeighborhoodBasedAlgorithm):

    def __init__(self, config):
        super().__init__("ItemBasedAlgorithm", config)

    def __neighborhood__(self):
        self._mean = ma.mean(self._rating, axis=1, keepdims=True)
        self._mean_center_rating = self._rating - self._mean

        assert self.config.sim_config.name in ["person", "discounted_person", "amplify_person", "idf_person", "pca_person"]

        similaritor = SimilaritorFactory(self.name, self.config)
        self._sim = similaritor(self._mean_center_rating.filled(0).T)
        sorted_neighborhood = ma.argsort(self._sim, axis=1, endwith=False)
        users_num, items_num = self._rating.shape

        self._neighborhood = []
        for i in range(users_num):
            neighborhood = []
            for j in range(items_num):
                if self.config.topk is not None:
                    neighborhood.append([el for el in sorted_neighborhood[i] if el != j and self._sim[j][el] is not ma.masked and self._rating[i][el] is not ma.masked][:-self.config.topk-1:-1])
                elif self.config.sim_threshold is not None:
                    neighborhood.append([el for el in sorted_neighborhood[i] if el != j and self._sim[j][el] > self.config.sim_threshold and self._rating[i][el] is not ma.masked])
                else:
                    raise RuntimeError("topk or sim_threshold must be setted")
            self._neighborhood.append(neighborhood)

    def __predict__(self):
        rating_hat = ma.masked_equal(np.zeros(self._rating.shape), 0)
        users_num, items_num = self._rating.shape
        start = time.clock()

        from recsys.algorithm.predictor import norm_predictor
        predictor = norm_predictor

        for i in range(users_num):
            if i % 10 == 0:
                print("[__predict__:{:.2f}s] {},{} {}%".format((time.clock()-start),i, users_num, i * 100 / users_num))
            for j in range(items_num):
                if self._neighborhood[i][j]:
                    continue

                rating_hat[i, j] =  predictor(rating=self._rating[i, self._neighborhood[i][j]],
                                              sim=self._sim[i, self._neighborhood[i][j]])

        return rating_hat
