#!/bin/env python
# -*- coding: utf-8 -*-

import math

import numpy as np
import numpy.ma as ma

from recsys.algorithm.similarity import pca_person
from recsys.algorithm.test_data import test_rating_data


def _simulate_np_cov(z1, z2, ddof=0):
    mean_z1 = np.mean(z1)
    mean_z2 = np.mean(z2)
    return np.dot(z1-mean_z1, z2-mean_z2) / (len(z1)- ddof)

def _simulate_np_var(z1, ddof=0):
    return _simulate_np_cov(z1, z1, ddof)

def _simulate_corrcoef_with_cov(covij, covii, covjj):
    return covij/math.sqrt(covii)/math.sqrt(covjj)

def _simulate_corrcoef_with_cov_matrix(cov):
    return _simulate_corrcoef_with_cov(cov[0,1], cov[0,0], cov[1,1])


class TestNumpyMethod(object):
    def test_var(self):
        for z in test_rating_data:
            assert math.isclose(np.var(z), _simulate_np_var(z), abs_tol=1e-6)

    def test_cov(self):
        for z1 in test_rating_data:
            for z2 in test_rating_data:
                assert math.isclose(np.cov(z1,z2, ddof=0)[0,1], _simulate_np_cov(z1,z2,ddof=0))
                assert math.isclose(np.cov(z1,z2)[0,1], _simulate_np_cov(z1,z2,ddof=1))

    def test_correcoef_with_diff_ddof(self):
        corrcoef = lambda cov: cov[0,1]/math.sqrt(cov[0,0]/math.sqrt(cov[1,1]))
        for i,z1 in enumerate(test_rating_data):
            for j,z2 in enumerate(test_rating_data):
                np_cov0 = np.cov(z1,z2,ddof=0)
                np_cov1 = np.cov(z1,z2,ddof=1)
                assert math.isclose(_simulate_corrcoef_with_cov_matrix(np_cov0), _simulate_corrcoef_with_cov_matrix(np_cov1))

    def test_corrcoef(self):
        np_coef = np.corrcoef(test_rating_data)
        for i,z1 in enumerate(test_rating_data):
            for j,z2 in enumerate(test_rating_data):
                np_cov0 = np.cov(z1,z2, ddof=0)
                assert math.isclose(np_coef[i,j], _simulate_corrcoef_with_cov_matrix(np_cov0))
                np_cov1 = np.cov(z1,z2, ddof=1)
                assert math.isclose(np_coef[i,j], _simulate_corrcoef_with_cov_matrix(np_cov1))


        for i,z1 in enumerate(test_rating_data):
            for j,z2 in enumerate(test_rating_data):
                cov12 = _simulate_np_cov(z1, z2, ddof=1)
                cov11 = _simulate_np_cov(z1, z1, ddof=1)
                cov22 = _simulate_np_cov(z2, z2, ddof=1)
                assert math.isclose(np_coef[i,j], _simulate_corrcoef_with_cov(cov12, cov11, cov22))
                cov12 = _simulate_np_cov(z1, z2, ddof=0)
                cov11 = _simulate_np_cov(z1, z1, ddof=0)
                cov22 = _simulate_np_cov(z2, z2, ddof=0)
                assert math.isclose(np_coef[i,j], _simulate_corrcoef_with_cov(cov12, cov11, cov22))


class TestCorrcoef(object):
    def test_corrcoef(self):
        r = ma.masked_equal(np.load("data/ml-1m/rating.npy"), 0)

        import pdb; pdb.set_trace()

        sim = ma.corrcoef(r[0], r[2412])
        print(sim)

        print(np.corrcoef(r[0].filled(0), r[2412].filled(0)))
        sim2 = ma.corrcoef(ma.vstack([r[0], r[2412]]))
        print(sim2)

        print(ma.dot(r[0], r[2412])/math.sqrt(ma.dot(r[0],r[0]))/math.sqrt(ma.dot(r[2412],r[2412])))

        r0_m = r[0] - ma.mean(r[0])
        r1_m = r[2412] - ma.mean(r[2412])
        print(ma.dot(r0_m, r1_m)/math.sqrt(ma.dot(r0_m,r0_m))/math.sqrt(ma.dot(r1_m,r1_m)))
