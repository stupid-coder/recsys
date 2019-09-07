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

def cosine(rating, **unused_kwargs):
    _filled_rating = rating.filled(0)
    _c = np.dot(_filled_rating, _filled_rating.T)
    _diag = np.sqrt(np.diag(_c))
    _denom = np.outer(_diag, _diag)
    return _c / _denom

def discounted_person(beta, cache_person):
    def _sim(rating, **unused_kwargs):
        p = cache_person(rating)
        count = np.zeros(rating.shape)
        count[~rating.mask] = 1
        count = np.dot(count, count.T)
        weight = np.fmin(count, beta) / beta
        sim = p * weight
        return sim
    return _sim

def amplify_person(alpha, cache_person):
    def _sim(rating, **unused_kwargs):
        p = cache_person(rating)
        return p ** alpha
    return _sim

def idf_person(rating):
    total = rating.shape[0]
    notmaskcount = np.sum(np.logical_not(ma.getmaskarray(rating)).astype(int), axis=0)
    weight = ma.masked_equal(ma.sqrt(ma.log(total / notmaskcount)), 0)
    w_rating = rating * weight
    return person(w_rating-ma.mean(w_rating, axis=1, keepdims=True))

def pca_person(dims):
    def _sim(ratings, **unused_kwargs):
        mean_ratings = ratings.mean(axis=0, keepdims=True)
        ratings = ratings - mean_ratings
        mean_ratings = ratings.mean(axis=1, keepdims=True)
        ratings = ratings - mean_ratings
        ratings = pca(ratings.filled(0),dims)
        return np.corrcoef(ratings)
    return _sim


def SimilaritorFactory(name, config):
    person_cache = cached("recsys/algorithm/cache/{}_person.npy".format(name))
    if config.sim_config.name == "person":
        return person_cache(person)
    elif config.sim_config.name == "discounted_person":
        return discounted_person(config.sim_config.discounted_beta, person_cache(person))
    elif config.sim_config.name == "amplify_person":
        return amplify_person(config.sim_config.amplify_alpha, person_cache(person))
    elif config.sim_config.name == "idf_person":
        return cached("recsys/algorithm/cache/{}_idf_person.npy".format(name))(idf_person)
    elif config.sim_config.name == "pca_person":
        return cached("recsys/algorithm/cache/{}_{}_pca_person.npy".format(name, config.sim_config.dims))(pca_person(config.sim_config.dims))
    elif config.sim_config.name == "cosine":
        return cached("recsys/algorithm/cache/{}_pca_cosine.npy".format(name))(cosine)
    else:
        raise NotImplementedError("[SimilaritorFactory] {} not implemented".format(config.sim_config.name))
