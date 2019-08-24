#!/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def pca(data, dims):
    mean_samples = np.mean(data, axis=0, keepdims=True)
    norm_data = data - mean_samples
    cov_matrix = np.dot(norm_data.T, norm_data)
    eig_val, eig_vec = np.linalg.eig(cov_matrix)
    eig_vec = eig_vec[:, np.argsort(eig_val)[::-1]]
    return np.dot(norm_data, eig_vec[:, :dims])
