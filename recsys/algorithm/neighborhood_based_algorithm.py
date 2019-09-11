#!/bin/env python
# -*- coding: utf-8
import logging
import time
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

    def __fit__(self, rating, row=True):
        if isinstance(rating, ma.MaskedArray):
            self._rating = rating
        else:
            self._rating = ma.masked_equal(rating, 0)

        self._mean = ma.mean(self._rating, axis=1, keepdims=True)
        self._sigma = ma.std(self._rating, axis=1, keepdims=True)
        self._mean_center_rating = self._rating - self._mean
        self._z = self._mean_center_rating / self._sigma

        assert self.config.sim_config.name in ["person", "discounted_person", "amplify_person", "idf_person", "pca_person","cosine"]

        similaritor = SimilaritorFactory(self.config.sim_config)

        if row:
            self._sim = similaritor(rating=self._rating, mean_center_rating=self._mean_center_rating)
        else:
            self._rating = self._rating.T
            self._mean_center_rating = self._mean_center_rating.T
            self._z = self._z.T
            self._sim = similaritor(rating=self._rating, mean_center_rating=self._mean_center_rating)

    def __predict__(self, row=True):

        rating_hat = ma.masked_equal(np.zeros(self._rating.shape), 0)
        m, n = self._rating.shape
        _predictor = PredictorFactory(self.config.predictor_config)

        self._sim[np.diag_indices(self._sim.shape[0])] = 0.0

        start = time.perf_counter()
        for j in range(n):
            if j % 10 == 0:
                logger.debug("[__predict__ {:.2f}s] {}/{}={:.2f}%".format(time.perf_counter()-start, j, n, j/n*100))

            _rating_m = np.where(self._rating[:, j] > 0)[0]
            _sim = self._sim[:, _rating_m]

            _rating = self._rating[_rating_m, j]
            _mean_center_rating = self._mean_center_rating[_rating_m, j]
            _z = self._z[_rating_m, j]

            if self.config.topk is not None:
                _neighborhood = np.argsort(_sim, axis=1)[:, -self.config.topk:]

                _num_n = _neighborhood.shape[1]
                _sim = _sim[[int(i/_num_n) for i in range(_neighborhood.size)], _neighborhood.flat].reshape((m, -1))
                _rating = _rating[_neighborhood.flat].reshape((m, -1))
                _mean_center_rating = _mean_center_rating[_neighborhood.flat].reshape((m, -1))
                _z = _z[_neighborhood.flat].reshape((m, -1))

                if _neighborhood.shape[1] < self.config.topk:
                    logger.warning("{} neighborhood {} < {}".format(j, _neighborhood.shape[1], self.config.topk))

                if _neighborhood.shape[1] <= 1:
                    continue

            elif self.config.sim_threshold is not None:
                _sim = ma.where(_sim > self.config.sim_threshold, _sim, 0)
                if 0 in ma.sum(_sim, axis=1):
                    logger.warning("{} no neighborhood".format(j))
            else:
                raise RuntimeError("topk or sim_threshold must be setted")

            rating_hat[:, j] = _predictor(rating=_rating,
                                          mean=np.squeeze(self._mean) if row else self._mean[j][0],
                                          sigma=np.squeeze(self._sigma) if row else self._sigma[j][0],
                                          mean_center_rating=_mean_center_rating,
                                          z=_z,
                                          sim=_sim)
        return rating_hat

class UserBasedNeighborhoodAlgorithm(NeighborhoodBasedAlgorithm):
    def __init__(self, config):
        super().__init__("UserBasedNeighborhoodAlgorithm", config)

    def fit(self, rating):
        self.__fit__(rating)

    def predict(self, rating=None):
        return self.__predict__()


class ItemBasedNeighborhoodAlgorithm(NeighborhoodBasedAlgorithm):
    def __init__(self, config):
        super().__init__("ItemBasedNeighborhoodAlgorithm", config)

    def fit(self, rating):
        self.__fit__(rating, row=False)

    def predict(self, rating=None):
        return self.__predict__(row=False).T
