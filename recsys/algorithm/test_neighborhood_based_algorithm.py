#!/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.ma as ma

from recsys.algorithm.neighborhood_based_algorithm import (ItemBasedNeighborhoodAlgorithm,
                                                           NeighborhoodBasedConfig,
                                                           PredictorConfig,
                                                           SimilarityConfig,
                                                           UserBasedNeighborhoodAlgorithm)
from recsys.algorithm.test_data import test_rating_data
from recsys.metric.metric import rmse


class TestUserBasedAlgorithm(object):

    def test_mean_center_predictor_algo(self):
        algo = UserBasedNeighborhoodAlgorithm(NeighborhoodBasedConfig(sim_config=SimilarityConfig(name="person"), topk=2, predictor_config=PredictorConfig(name="mean_center")))
        algo.fit(test_rating_data)
        rating_hat = algo.predict(None)
        print(rating_hat)
        print(rmse(rating_hat, ma.masked_equal(test_rating_data, 0)))  # 0.626923301909203



    def test_mean_center_predictor_algo_with_sim_threshold(self):
        algo = UserBasedNeighborhoodAlgorithm(NeighborhoodBasedConfig(sim_config=SimilarityConfig(name="person", discounted_beta=None), topk=None, sim_threshold=0.0, predictor_config=PredictorConfig(name="mean_center")))
        algo.fit(test_rating_data)
        rating_hat = algo.predict(None)
        print(rating_hat)
        print(rmse(rating_hat, ma.masked_equal(test_rating_data, 0)))  # 0.6314698247230878


    def test_norm_predictor_algo(self):
        algo = UserBasedNeighborhoodAlgorithm(NeighborhoodBasedConfig(sim_config=SimilarityConfig(name="person", discounted_beta=None), topk=2, sim_threshold=None, predictor_config=PredictorConfig(name="norm")))
        algo.fit(test_rating_data)
        rating_hat = algo.predict(None)
        print(rating_hat)
        print(rmse(rating_hat, ma.masked_equal(test_rating_data, 0)))  # 2.125021074648673


    def test_discounted_sim_algo(self):
        algo = UserBasedNeighborhoodAlgorithm(NeighborhoodBasedConfig(sim_config=SimilarityConfig(name="discounted_person", discounted_beta=6), topk=2, sim_threshold=None, predictor_config=PredictorConfig(name="mean_center")))
        algo.fit(test_rating_data)
        rating_hat = algo.predict(None)
        print(rating_hat)
        print(rmse(rating_hat, ma.masked_equal(test_rating_data, 0)))  # 0.642910168586099


    def test_mean_sigma_predictor_algo(self):
        algo = UserBasedNeighborhoodAlgorithm(NeighborhoodBasedConfig(sim_config=SimilarityConfig(name="discounted_person", discounted_beta=6), topk=2, sim_threshold=None, predictor_config=PredictorConfig(name="mean_sigma")))
        algo.fit(test_rating_data)
        rating_hat = algo.predict(None)
        print(rating_hat)
        print(rmse(rating_hat, ma.masked_equal(test_rating_data, 0)))  # 0.6249970642035232


    def test_amplify_sim_algo(self):
        algo = UserBasedNeighborhoodAlgorithm(NeighborhoodBasedConfig(sim_config=SimilarityConfig(name="amplify_person", amplify_alpha=3), topk=2, sim_threshold=None, predictor_config=PredictorConfig(name="mean_sigma")))
        algo.fit(test_rating_data)
        rating_hat = algo.predict(None)
        print(rating_hat)
        print(rmse(rating_hat, ma.masked_equal(test_rating_data, 0)))  # 0.5921262449508355

    def test_idf_sim_algo(self):
        algo = UserBasedNeighborhoodAlgorithm(NeighborhoodBasedConfig(sim_config=SimilarityConfig(name="idf_person"), topk=2, sim_threshold=None, predictor_config=PredictorConfig(name="mean_center")))
        algo.fit(test_rating_data)
        rating_hat = algo.predict(None)
        print(rating_hat)
        print(rmse(rating_hat, ma.masked_equal(test_rating_data, 0)))  # 0.9546439193233666

    def test_pca_person_sim_algo(self):
        algo = UserBasedNeighborhoodAlgorithm(NeighborhoodBasedConfig(sim_config=SimilarityConfig(name="pca_person", dims=2), topk=2, sim_threshold=None, predictor_config=PredictorConfig(name="mean_center")))
        algo.fit(test_rating_data)
        rating_hat = algo.predict(None)
        print(rating_hat)
        print(rmse(rating_hat, ma.masked_equal(test_rating_data, 0)))  # 0.6237294778495414


class TestItemBasedAlgorithm(object):
    def test_mean_center_predictor_algo(self):
        algo = ItemBasedNeighborhoodAlgorithm(NeighborhoodBasedConfig(sim_config=SimilarityConfig(name="cosine"), topk=2, sim_threshold=None, predictor_config=PredictorConfig(name="mean_center")))
        algo.fit(test_rating_data)
        rating_hat = algo.predict(None)

        print(rating_hat)
        print(rmse(rating_hat, ma.masked_equal(test_rating_data, 0)))  # 0.6543287113746402

    def test_pca_person_sim_algo(self):
        algo = ItemBasedNeighborhoodAlgorithm(NeighborhoodBasedConfig(sim_config=SimilarityConfig(name="pca_person", dims=2), topk=2, sim_threshold=None, predictor_config=PredictorConfig(name="norm")))
        algo.fit(test_rating_data)
        rating_hat = algo.predict(None)
        print(rating_hat)
        print(rmse(rating_hat, ma.masked_equal(test_rating_data, 0)))  # 0.6237294778495414
