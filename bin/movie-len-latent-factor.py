#!/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import logging.config
import sys

import numpy as np
import numpy.ma as ma

from recsys.algorithm.latent_factor_model import (LatentFactorConfig,
                                                  UnconstrainedMatrixFactorAlgorithm)
from recsys.dataset.dataset import MovieLenDataset
from recsys.metric import metric

logging.config.fileConfig("conf/logging.conf", disable_existing_loggers=False)

logger = logging.getLogger("main")

parser = argparse.ArgumentParser(description="movie len algorithm")
parser.add_argument("--algo", type=str, help="algorithm [umf, ]")
parser.add_argument("--dataset", type=str, default="data/ml-1m", help="directory of movie len")
parser.add_argument("--embedding_size", type=int, default=50, help="embedding size of latent factor")
parser.add_argument("--learn_rate", type=float, default=0.001, help="learning rate about regression model")
parser.add_argument("--epochs", type=int, default=1000, help="epochs about regression model")
parser.add_argument("--model_dir", type=str, help="model directory to save and restore")


if __name__ == "__main__":
    args = parser.parse_args()
    logger.info("--------------------------------------------------")
    print("args: {}".format(args))
    ml = MovieLenDataset(args.dataset)

    if args.algo == "umf":
        config = LatentFactorConfig(model_dir=args.model_dir, embedding_size=args.embedding_size, epochs=args.epochs, learn_rate=args.learn_rate)
        algorithm = UnconstrainedMatrixFactorAlgorithm(config)
    else:
        print("[USAGE] algo must be in [umf, ]")
        sys.exit(-1)

    algorithm.fit(ml.R)
    hat_rating = algorithm.predict(None)
    rating = ma.masked_equal(ml.R, 0)

    logger.info("|{}|{}|".format(args, metric.rmse(rating[:, 0], hat_rating[:, 0])))
