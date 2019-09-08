#!/bin/env python
# -*- coding: utf-8
import logging
import time
from multiprocessing import Pool, cpu_count
from typing import NamedTuple

import numpy as np
from numpy import ma

from recsys.algorithm.algorithm import Algorithm
from recsys.algorithm.predictor import PredictorFactory
from recsys.algorithm.similarity import SimilaritorFactory

logger = logging.getLogger(__name__)

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

    def __do_neighborhood__(self, ui_pair):
        i, j = ui_pair
        if self.config.topk is not None:
            return (i, j, [el for el in self._sorted_neighborhood[i] if el != i and self._sim[i][el] is not ma.masked and self._rating[el][j] is not ma.masked][:-self.config.topk-1:-1])

        if self.config.sim_threshold is not None:
            return (i, j, [el for el in self._sorted_neighborhood[i] if el != i and self._sim[i][el] > self.config.sim_threshold and self._rating[el][j] is not ma.masked])

        raise RuntimeError("topk or sim_threshold must be setted")

    def __neighborhood__(self):
        self._mean = ma.mean(self._rating, axis=1, keepdims=True)
        self._sigma = ma.std(self._rating, axis=1, keepdims=True)
        self._mean_center_rating = self._rating - self._mean
        self._z = self._mean_center_rating / self._sigma

        assert self.config.sim_config.name in ["person", "discounted_person", "amplify_person", "idf_person", "pca_person"]

        similaritor = SimilaritorFactory(self.name, self.config)

        logger.info("[__neighborhood__:{:.2f}s] calculate neighborhood begin".format(time.perf_counter()))
        self._sim = similaritor(rating=self._rating, mean_center_rating=self._mean_center_rating)
        logger.info("[__neighborhood__:{:.2f}s] calculate neighborhood end".format(time.perf_counter()))
        self._sorted_neighborhood = ma.argsort(self._sim, axis=1, endwith=False)
        users_num, items_num = self._rating.shape

        with Pool(int(cpu_count()*0.8)) as p:
            self._neighborhood = [[[] for j in range(items_num)] for i in range(users_num)]
            for (i, j, neighborhood) in p.map(self.__do_neighborhood__, [(i, j) for i in range(users_num) for j in range(items_num)]):
                self._neighborhood[i][j] = neighborhood

    def __do_predict__(self, ui_pair):
        i, j = ui_pair
        if not self._neighborhood[i][j]:
            return (i, j, ma.masked)
        return (i, j, self._predictor(rating=self._rating[self._neighborhood[i][j], j],
                                      mean=self._mean[i],
                                      sigma=self._sigma[i],
                                      mean_center_rating=self._mean_center_rating[self._neighborhood[i][j], j],
                                      z=self._z[self._neighborhood[i][j], j],
                                      sim=self._sim[i, self._neighborhood[i][j]]))

    def __predict__(self):
        rating_hat = ma.masked_equal(np.zeros(self._rating.shape), 0)
        users_num, items_num = self._rating.shape
        self._predictor = PredictorFactory(self.config.predictor_config)

        with Pool(int(cpu_count() * 0.8)) as p:
            for (i, j, r_hat) in p.map(self.__do_predict__, [(i, j) for i in range(users_num) for j in range(items_num)]):
                rating_hat[i,j] = r_hat

        return rating_hat


class ItemBasedAlgorithm(NeighborhoodBasedAlgorithm):

    def __init__(self, config):
        super().__init__("ItemBasedAlgorithm", config)

    def __do_neighborhood__(self, ui_pair):
        i,j = ui_pair
        if self.config.topk is not None:
            return (i, j, [el for el in self._sorted_neighborhood[j] if el != j and self._sim[j][el] is not ma.masked and self._rating[i][el] is not ma.masked][:-self.config.topk-1:-1])
        if self.config.sim_threshold is not None:
            return (i, j, [el for el in self._sorted_neighborhood[j] if el != j and self._sim[j][el] > self.config.sim_threshold and self._rating[i][el] is not ma.masked])
        raise RuntimeError("topk or sim_threshold must be setted")

    def __neighborhood__(self):
        self._mean = ma.mean(self._rating, axis=1, keepdims=True)
        self._mean_center_rating = self._rating - self._mean

        assert self.config.sim_config.name in ["cosine"]

        similaritor = SimilaritorFactory(self.name, self.config)
        logger.info("[__neighborhood__:{:.2f}s] calculate neighborhood begin".format(time.perf_counter()))
        self._sim = similaritor(rating=self._mean_center_rating.T)
        logger.info("[__neighborhood__:{:.2f}s] calculate neighborhood begin".format(time.perf_counter()))
        self._sorted_neighborhood = ma.argsort(self._sim, axis=1, endwith=False)
        users_num, items_num = self._rating.shape


        with Pool(int(cpu_count()*0.8)) as p:
            self._neighborhood = [[[] for j in range(items_num)] for i in range(users_num)]
            for (i,j,neighborhood) in p.map(self.__do_neighborhood__, [(i, j) for i in range(users_num) for j in range(items_num)]):
                self._neighborhood[i][j] = neighborhood

    def __do_predict__(self, ui_pair):
        i, j = ui_pair
        if not self._neighborhood[i][j]:
            return (i, j, ma.masked)
        return (i, j, self._predictor(rating=self._rating[i, self._neighborhood[i][j]],
                                      sim=self._sim[i, self._neighborhood[i][j]]))

    def __predict__(self):
        rating_hat = ma.masked_equal(np.zeros(self._rating.shape), 0)
        users_num, items_num = self._rating.shape

        from recsys.algorithm.predictor import norm_predictor
        self._predictor = norm_predictor

        with Pool(int(cpu_count()*0.8)) as p:
            for (i, j, r_hat) in p.map(self.__do_predict__, [(i, j) for i in range(users_num) for j in range(items_num)]):
                rating_hat[i, j] = r_hat
        return rating_hat
