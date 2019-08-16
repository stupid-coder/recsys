#!/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from recsys.algorithm.neighborhood_based_algorithm import (ItemBasedAlgorithm,
                                                           NeighborhoodBasedConfig,
                                                           PredictorConfig,
                                                           SimilarityConfig,
                                                           UserBasedAlgorithm)
from recsys.algorithm.test_data import test_rating_data


class TestUserBasedAlgorithm(object):

    def test_mean_center_predictor_algo(self):
        algo = UserBasedAlgorithm(NeighborhoodBasedConfig(sim_config=SimilarityConfig(name="person", discounted_beta=None), topk=2, sim_threshold=None, predictor_config=PredictorConfig(name="mean_center")))
        algo.fit(test_rating_data)
        rating_hat = algo.predict(None)
        print(rating_hat)

    def test_mean_center_predictor_algo_with_sim_threshold(self):
        algo = UserBasedAlgorithm(NeighborhoodBasedConfig(sim_config=SimilarityConfig(name="person", discounted_beta=None), topk=None, sim_threshold=0.0, predictor_config=PredictorConfig(name="mean_center")))
        algo.fit(test_rating_data)
        rating_hat = algo.predict(None)
        print(rating_hat)


    def test_norm_predictor_algo(self):
        algo = UserBasedAlgorithm(NeighborhoodBasedConfig(sim_config=SimilarityConfig(name="person", discounted_beta=None), topk=2, sim_threshold=None, predictor_config=PredictorConfig(name="norm")))
        algo.fit(test_rating_data)
        rating_hat = algo.predict(None)
        print(rating_hat)

    def test_discounted_sim_algo(self):
        algo = UserBasedAlgorithm(NeighborhoodBasedConfig(sim_config=SimilarityConfig(name="discounted_person", discounted_beta=6), topk=2, sim_threshold=None, predictor_config=PredictorConfig(name="mean_center")))
        algo.fit(test_rating_data)
        rating_hat = algo.predict(None)
        print(rating_hat)


    def test_mean_sigma_predictor_algo(self):
        algo = UserBasedAlgorithm(NeighborhoodBasedConfig(sim_config=SimilarityConfig(name="discounted_person", discounted_beta=6), topk=2, sim_threshold=None, predictor_config=PredictorConfig(name="mean_sigma")))
        algo.fit(test_rating_data)
        rating_hat = algo.predict(None)
        print(rating_hat)

    def test_amplify_sim_algo(self):
        algo = UserBasedAlgorithm(NeighborhoodBasedConfig(sim_config=SimilarityConfig(name="amplify_person", amplify_alpha=3), topk=2, sim_threshold=None, predictor_config=PredictorConfig(name="mean_sigma")))
        algo.fit(test_rating_data)
        rating_hat = algo.predict(None)
        print(rating_hat)

    def test_idf_sim_algo(self):
        algo = UserBasedAlgorithm(NeighborhoodBasedConfig(sim_config=SimilarityConfig(name="idf_person"), topk=2, sim_threshold=None, predictor_config=PredictorConfig(name="mean_center")))
        algo.fit(test_rating_data)
        rating_hat = algo.predict(None)
        print(rating_hat)


class TestItemBasedAlgorithm(object):
    def testE_mean_center_predictor_algo(self):
        algo = ItemBasedAlgorithm(NeighborhoodBasedConfig(sim_config=SimilarityConfig(name="person"), topk=2, sim_threshold=None, predictor_config=PredictorConfig(name="mean_center")))
        algo.fit(test_rating_data)
        rating_hat = algo.predict(None)
        print(rating_hat)