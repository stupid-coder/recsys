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
parser.add_argument("--dir", type=str, help="directory of movie len")


if __name__ == "__main__":
    args = parser.parse_args()

    ml = MovieLenDataset(args.dir)
    algorithm = UserBasedAlgorithm(NeighborhoodBasedConfig(sim_config=SimilarityConfig(name="person"), topk=50, sim_threshold=None, predictor_config=PredictorConfig(name="mean_center")))
    algorithm.fit(ml.R)
    hat_rating = algorithm.predict(None)
    print("--------------------------------------------------")
    print("rmse:{}".format(metric.rmse(hat_rating, ma.masked_equal(ml.R, 0))))
