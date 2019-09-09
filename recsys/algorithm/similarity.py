#!/bin/env python
# -*- coding: utf-8 -*-
import math
import os

import numpy as np
import numpy.ma as ma

from recsys.common.cache import cached
from recsys.common.pca import pca


def person(mean_center_rating, **unused_kwargs):
    _filled_mean_center_rating = mean_center_rating.filled(0)
    _c = np.dot(_filled_mean_center_rating, _filled_mean_center_rating.T)
    _diag = np.sqrt(np.diag(_c))
    _denom = np.outer(_diag, _diag)
    return _c / _denom


def cosine(mean_center_rating, **unused_kwargs):
    _filled_rating = mean_center_rating.filled(0)
    _c = np.dot(_filled_rating, _filled_rating.T)
    _diag = np.sqrt(np.diag(_c))
    _denom = np.outer(_diag, _diag)
    return _c / _denom


def discounted_person(beta):
    def _sim(mean_center_rating, **unused_kwargs):
        p = person(mean_center_rating)
        count = ma.logical_not(ma.getmaskarray(mean_center_rating)).astype(np.int)
        count = np.dot(count, count.T)
        weight = np.fmin(count, beta) / beta
        sim = p * weight
        return sim
    return _sim


def amplify_person(alpha):
    def _sim(mean_center_rating, **unused_kwargs):
        p = person(mean_center_rating)
        return np.power(p, alpha)
    return _sim


def idf_person(rating, **unused_kwargs):
    total = rating.shape[0]
    notmaskcount = np.sum(np.logical_not(ma.getmaskarray(rating)).astype(int), axis=0)
    weight = ma.sqrt(ma.log(total / notmaskcount))
    w_rating = rating * weight
    return person(w_rating-ma.mean(w_rating, axis=1, keepdims=True))


def pca_person(dims):
    def _sim(rating, **unused_kwargs):
        mean_rating = rating.mean(axis=0, keepdims=True)
        rating = rating - mean_rating
        mean_rating = rating.mean(axis=1, keepdims=True)
        rating = rating - mean_rating
        rating = pca(rating.filled(0), dims)
        return np.corrcoef(rating)
    return _sim


def SimilaritorFactory(name, config):
    if config.sim_config.name == "person":
        return person
    elif config.sim_config.name == "discounted_person":
        return discounted_person(config.sim_config.discounted_beta)
    elif config.sim_config.name == "amplify_person":
        return amplify_person(config.sim_config.amplify_alpha)
    elif config.sim_config.name == "idf_person":
        return idf_person
    elif config.sim_config.name == "pca_person":
        return pca_person(config.sim_config.dims)
    elif config.sim_config.name == "cosine":
        return cosine
    else:
        raise NotImplementedError("[SimilaritorFactory] {} not implemented".format(config.sim_config.name))
