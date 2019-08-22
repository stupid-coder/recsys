#!/bin/env python
# -*- coding: utf-8 -*-

import argparse

import numpy as np
import numpy.ma as ma

from recsys.algorithm.neighborhood_based_algorithm import (NeighborhoodBasedConfig,
                                                           PredictorConfig,
                                                           SimilarityConfig,
                                                           UserBasedAlgorithm)
from recsys.dataset.dataset import MovieLenDataset
from recsys.metric import metric

parser = argparse.ArgumentParser(description="movie len item-based algorithm")
parser.add_argument("dataset", type=str, help="directory of movie len")
parser.add_argument("topk", type=int, help="topk similarity to use")
parser.add_argument("sim", type=str, help="similaritor")
parser.add_argument("predictor", type=str, help="predictor")
parser.add_argument("--discounted_beta", type=int, help="discounted beta for weight")
parser.add_argument("--amplify_alpha", type=float, help="amplify alpha for similarity")
parser.add_argument("--sim_threshold", type=float, help="similarity threshold")


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    ml = MovieLenDataset(args.dataset)
    algorithm = UserBasedAlgorithm(NeighborhoodBasedConfig(sim_config=SimilarityConfig(name=args.sim, discounted_beta=args.discounted_beta, amplify_alpha=args.amplify_alpha), topk=args.topk, sim_threshold=args.sim_threshold, predictor_config=PredictorConfig(name=args.predictor)))
    algorithm.fit(ml.R)
    hat_rating = algorithm.predict(None)
    print("--------------------------------------------------")
    print("rmse:{}".format(metric.rmse(hat_rating, ma.masked_equal(ml.R, 0))))
