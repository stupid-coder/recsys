#!/bin/env python
# -*- coding: utf-8 -*-
import math
import os

import numpy as np
import numpy.ma as ma


def person(ratings, **unused_kwargs):
    cache_name = os.path.join("recsys/algorithm/cache", "person.npy")
    if os.path.exits(cache_name):
        return np.load(open(cache_name, "rb"))
    else:
        sim = ma.corrcoef(ratings)
        np.save(open(cache_name, "wb"), sim)
        return sim

def discounted_person(beta):
    def _sim(ratings, **unused_kwargs):
        cache_name = os.path.join("recsys/algorithm/cache", "discounted_person.{}.npy".format(beta))
        if os.path.exists(cache_name):
            return np.load(open(cache_name, "rb"))
        else:
            p = person(ratings)
            count = np.zeros(ratings.shape)
            count[~ratings.mask] = 1
            count = np.dot(count, count.T)
            weight = np.fmin(count, beta) / beta
            sim = p * weight
            np.save(open(cache_name, "wb"), sim)
            return sim
    return _sim

def amplify_person(alpha):
    def _sim(ratings, **unused_kwargs):
        p = person(ratings)
        return p ** alpha
    return _sim


def idf_person(ratings):
    cache_name = os.path.join("recsys/algorithm/cache", "idf_person.npy")
    if os.path.exists(cache_name):
        return np.load(open(cache_name, "rb"))
    else:
        total = ratings.shape[0]
        notmaskcount = np.sum(np.logical_not(ma.getmaskarray(ratings)).astype(int), axis=0)
        weight = ma.masked_equal(ma.sqrt(ma.log(total / notmaskcount)), 0)
        sim = ma.corrcoef(ratings * weight)
        np.save(open(cache_name, "wb"), sim)
        return sim


def SimilaritorFactory(sim_config):
    if sim_config.name == "person":
        return person
    elif sim_config.name == "discounted_person":
        return discounted_person(sim_config.discounted_beta)
    elif sim_config.name == "amplify_person":
        return amplify_person(sim_config.amplify_alpha)
    elif sim_config.name == "idf_person":
        return idf_person
    else:
        raise NotImplementedError("[SimilaritorFactory] {} not implemented".format(sim_config))
