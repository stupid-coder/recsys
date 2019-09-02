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

    def fit(self, rating):
        raise NotImplementedError("[{}] fit not implemented".format(self.__class__))

    def predict(self, rating):
        raise NotImplementedError("[{}] predict not implemented".format(self.__class__))
