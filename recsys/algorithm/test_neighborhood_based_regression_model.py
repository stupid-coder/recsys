#!/bin/env python

import logging
import sys

from numpy import ma

from recsys.algorithm.neighborhood_based_algorithm_regression_model import (ItemBasedRegressionModel,
                                                                            RegressionModelNeighborhoodBasedConfig,
                                                                            UserBasedRegressionModel)
from recsys.algorithm.test_data import test_rating_data
from recsys.metric.metric import rmse, statistic

logging.basicConfig(stream=sys.stdout)

class TestRegressionModel(object):

    def test_user_based_regression_model(self):
        algo = UserBasedRegressionModel(RegressionModelNeighborhoodBasedConfig(topk=3, lr=0.005, epochs=200, wdecay=0.0001))
        algo.fit(test_rating_data)
        hat_rating = algo.predict()
        print(statistic(ma.masked_equal(test_rating_data, 0), hat_rating))
        print(rmse(ma.masked_equal(test_rating_data, 0), hat_rating))
        print(algo.__parameters__())

    def test_item_based_regression_model(self):
        algo = ItemBasedRegressionModel(RegressionModelNeighborhoodBasedConfig(topk=3, lr=0.005, epochs=200, wdecay=0.0001))
        algo.fit(test_rating_data)
        hat_rating= algo.predict()
        print(statistic(ma.masked_equal(test_rating_data, 0), hat_rating))
        print(rmse(ma.masked_equal(test_rating_data, 0), hat_rating))
        print(algo.__parameters__())
