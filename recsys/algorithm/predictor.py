#!/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import numpy.ma as ma


def norm_predictor(rating, sim, **not_used_kwargs):
    return np.sum(rating * sim, axis=1) / np.sum(np.abs(sim), axis=1)

def mean_center_predictor(mean, mean_center_rating, sim, **not_used_kwargs):
    return mean + np.sum(mean_center_rating * sim, axis=1) / np.sum(np.abs(sim), axis=1)

def mean_sigma_predictor(mean, sigma, z, sim, **not_used_kwargs):
    return mean + sigma * ma.sum(sim * z, axis=1) / ma.sum(ma.abs(sim), axis=1)


def PredictorFactory(predictor_config):
    if predictor_config.name == "norm":
        return norm_predictor

    if predictor_config.name == "mean_center":
        return mean_center_predictor

    if predictor_config.name == "mean_sigma":
        return mean_sigma_predictor

    raise NotImplementedError("[PredictorFactory] {} not implemented".format(predictor_config))
