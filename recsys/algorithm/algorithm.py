#!/bin/env python
# -*- coding: utf-8 -*-

class Algorithm(object):
    def __init__(self, name, config):
        self._name = name
        self._config = config

    @property
    def config(self):
        return self._config

    @property
    def name(self):
        return self._name

    @property
    def sim(self):
        return self._sim

    @property
    def rating(self):
        return self._rating

    @property
    def mean(self):
        return self._mean

    @property
    def sigma(self):
        return self._sigma

    @property
    def mean_center_rating(self):
        return self._mean_center_rating

    @property
    def z(self):
        return self._z

    def fit(self, rating):
        raise NotImplementedError("[{}] fit not implemented".format(self.__class__))

    def predict(self, rating):
        raise NotImplementedError("[{}] predict not implemented".format(self.__class__))
