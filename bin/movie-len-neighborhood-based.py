#!/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import logging.config
import sys

import numpy as np
import numpy.ma as ma

from recsys.algorithm.neighborhood_based_algorithm import (ItemBasedNeighborhoodAlgorithm,
                                                           NeighborhoodBasedConfig,
                                                           PredictorConfig,
                                                           SimilarityConfig,
                                                           UserBasedNeighborhoodAlgorithm)
from recsys.algorithm.neighborhood_based_algorithm_regression_model import (ItemBasedRegressionModel,
                                                                            RegressionModelNeighborhoodBasedConfig,
                                                                            UserBasedRegressionModel)
from recsys.dataset.dataset import MovieLenDataset
from recsys.metric import metric

logging.config.fileConfig("conf/logging.conf", disable_existing_loggers=False)

logger = logging.getLogger("main")

parser = argparse.ArgumentParser(description="movie len item-based algorithm")
parser.add_argument("--algo", type=str, help="algorithm [user or item]")
parser.add_argument("--dataset", type=str, default="data/ml-1m", help="directory of movie len")
parser.add_argument("--sim", type=str, default="person", help="similaritor")
parser.add_argument("--predictor", type=str, default="mean_center", help="predictor")
parser.add_argument("--discounted_beta", type=int, help="discounted beta for weight")
parser.add_argument("--amplify_alpha", type=float, help="amplify alpha for similarity")
parser.add_argument("--sim_threshold", type=float, help="similarity threshold")
parser.add_argument("--topk", type=int, help="topk similarity to use")
parser.add_argument("--dims", type=int, help="reduce dimensions")
parser.add_argument("--learn_rate", type=float, default=0.001, help="learning rate about regression model")
parser.add_argument("--epochs", type=int, default=10000, help="epochs about regression model")
parser.add_argument("--weight_decay", type=float, default=0.001, help="weight decay or l2 regularization")
parser.add_argument("--check_gradient", type=bool, default=False, help="check gradient in optimization")


if __name__ == "__main__":
    args = parser.parse_args()
    logger.info("--------------------------------------------------")

    ml = MovieLenDataset(args.dataset)

    if args.algo == "user":
        algorithm = UserBasedNeighborhoodAlgorithm(NeighborhoodBasedConfig(sim_config=SimilarityConfig(name=args.sim, discounted_beta=args.discounted_beta, amplify_alpha=args.amplify_alpha, dims=args.dims), topk=args.topk, sim_threshold=args.sim_threshold, predictor_config=PredictorConfig(name=args.predictor)))
    elif args.algo == "item":
        algorithm = ItemBasedNeighborhoodAlgorithm(NeighborhoodBasedConfig(sim_config=SimilarityConfig(name=args.sim, discounted_beta=args.discounted_beta, amplify_alpha=args.amplify_alpha, dims=args.dims), topk=args.topk, sim_threshold=args.sim_threshold, predictor_config=PredictorConfig(name=args.predictor)))
    elif args.algo == "user_reg":
        algorithm = UserBasedRegressionModel(RegressionModelNeighborhoodBasedConfig(topk=args.topk, lr=args.learn_rate, epochs=args.epochs, wdecay=args.weight_decay, check_gradient=args.check_gradient))
    elif args.algo == "item_reg":
        algorithm = ItemBasedRegressionModel(RegressionModelNeighborhoodBasedConfig(topk=args.topk, lr=args.learn_rate, epochs=args.epochs, wdecay=args.weight_decay, check_gradient=args.check_gradient))
    else:
        print("[USAGE] algo must be in [user,item]")
    algorithm.fit(ml.R)
    hat_rating = algorithm.predict(None)
    rating = ma.masked_equal(ml.R, 0)

    logger.debug("rating_count: {rating_count}\tpredict_count:{predict_count}\tpredict_rate:{predict_rate}".format(**metric.statistic(rating, hat_rating)))
    logger.info("|{}|{}|".format(args, metric.rmse(rating, hat_rating)))
