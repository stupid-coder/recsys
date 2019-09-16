#!/bin/env python
# -*- coding: utf-8

import logging
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy import ma

from recsys.algorithm.algorithm import Algorithm

logger =logging.getLogger(__name__)

class LatentFactorConfig(NamedTuple):
    model_dir: str
    embedding_size: int = 50
    epochs: int = 1000
    learn_rate: float = 1e-3
    save_per_epochs: int = 5


class UnconstrainedMatrixFactorAlgorithm(Algorithm):
    """
    \hat{R} = UV^{T}

    Minimize || \hat{R} - R ||
    subject to:
    no constrained on U and V
    """
    def __init__(self, config):
        super().__init__("UnconstrainedMatrixFactorAlgorithm", config)
        self._epoch = 0
        assert self.config.model_dir

        model_path = Path(self.config.model_dir)

        if not model_path.exists():
            model_path.mkdir()
        else:
            model_file = None
            for mfile in model_path.glob("{}.[0-9]*".format(self.name)):
                mname, epoch, _ = mfile.name.split(".")
                if int(epoch) > self._epoch:
                    self._epoch = int(epoch)
                    model_file = mfile
            if model_file is not None:
                self.load("{}/{}".format(self.config.model_dir, model_file.name))

    def __forward__(self):
        return np.dot(self._U, self._V.T)

    def __backward__(self, hat_rating):
        _e = hat_rating - self._rating
        _g_U = ma.dot(_e, self._V) / self._rating.shape[0]
        _g_V = ma.dot(_e.T, self._U) / self._rating.shape[1]
        return _g_U, _g_V

    def __loss__(self, hat_rating):
        return 0.5 * ma.mean(ma.power(hat_rating - self._rating, 2))

    def fit(self, rating):
        if isinstance(rating, ma.MaskedArray):
            self._rating = rating
        else:
            self._rating = ma.masked_equal(rating, 0)

        self._user_mean_rating = ma.mean(self._rating, axis=1)
        self._item_mean_rating = ma.mean(self._rating, axis=0)

        if "_U" not in self.__dict__:
            self._U = np.random.randn(self._rating.shape[0], self.config.embedding_size)

        if "_V" not in self.__dict__:
            self._V = np.random.randn(self._rating.shape[1], self.config.embedding_size)

        start = time.perf_counter()
        for epoch in range(self._epoch, self.config.epochs):
            self._epoch = epoch

            _hat_rating = self.__forward__()

            _loss = self.__loss__(_hat_rating)

            logger.debug("[{:4d} epoch\t{:.2f}s] loss:{:.2f}".format(self._epoch, time.perf_counter()-start,_loss))

            _g_U, _g_V = self.__backward__(_hat_rating)

            self._U -= self.config.learn_rate * _g_U
            self._V -= self.config.learn_rate * _g_V

            if self._epoch % self.config.save_per_epochs == 0:
                self.save()

        if self._epoch % self.config.save_per_epochs != 0:
            self.save()

    def predict(self, rating):
        return self.__forward__()

    def save(self):
        np.savez("{}/{}.{}".format(self.config.model_dir, self.name, self._epoch), U=self._U, V=self._V)

    def load(self, model_file):
        parameters = np.load(open(model_file, "rb"))
        self._U = parameters["U"]
        self._V = parameters["V"]
