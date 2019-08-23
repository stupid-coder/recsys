#!/bin/env python
# -*- coding: utf-8 -*-
import math
import os

import numpy as np
import numpy.ma as ma

from recsys.common.cache import cached


def person(ratings, **unused_kwargs):
    return ma.corrcoef(ratings)

def discounted_person(beta, cache_person):
    def _sim(ratings, **unused_kwargs):
        p = cache_person(ratings)
        count = np.zeros(ratings.shape)
        count[~ratings.mask] = 1
        count = np.dot(count, count.T)
        weight = np.fmin(count, beta) / beta
        sim = p * weight
        return sim
    return _sim

def amplify_person(alpha, cache_person):
    def _sim(ratings, **unused_kwargs):
        p = cache_person(ratings)
        return p ** alpha
    return _sim

def idf_person(ratings):
    total = ratings.shape[0]
    notmaskcount = np.sum(np.logical_not(ma.getmaskarray(ratings)).astype(int), axis=0)
    weight = ma.masked_equal(ma.sqrt(ma.log(total / notmaskcount)), 0)
    sim = ma.corrcoef(ratings * weight)
    return sim


def SimilaritorFactory(config):
    person_cache = cached("recsys/algorithm/cache/{}_person.npy".format(self.name))
    if config.sim_config.name == "person":
        return person_cache(person)
    elif sim_config.name == "discounted_person":
        return discounted_person(sim_config.discounted_beta, person_cache(person))
    elif sim_config.name == "amplify_person":
        return cache(amplify_person(sim_config.amplify_alpha, person_cache(person)))
    elif sim_config.name == "idf_person":
        return cached("recsys/algorithm/cache/{}_idf_person.npy".format(self.name))(idf_person)
    else:
        raise NotImplementedError("[SimilaritorFactory] {} not implemented".format(sim_config))
