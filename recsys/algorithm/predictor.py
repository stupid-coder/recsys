#!/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import numpy.ma as ma


def norm_predictor(rating, sim, **not_used_kwargs):
    return ma.sum(sim * rating, axis=0) / ma.sum(ma.abs(sim))


def mean_center_predictor(mean, mean_center_rating, sim, **not_used_kwargs):
    return mean + ma.sum(sim * mean_center_rating, axis=0) / ma.sum(ma.abs(sim))

def mean_sigma_predictor(mean, sigma, z, sim, **not_used_kwargs):
    return mean + sigma * ma.sum(sim * z, axis=0) / ma.sum(ma.abs(sim))


def PredictorFactory(predictor_config):
    if predictor_config.name == "norm":
        return norm_predictor
    elif predictor_config.name == "mean_center":
        return mean_center_predictor
    elif predictor_config.name == "mean_sigma":
        return mean_sigma_predictor
    else:
        raise NotImplementedError("[PredictorFactory] {} not implemented".format(predictor_config))