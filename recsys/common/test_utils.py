#!/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

from recsys.common import utils


def test_get_keywords_dict():
    s = pd.Series(["title test (123)", "test1, test (234)"])
    movie_pd = pd.DataFrame(s, columns=["title"])
    keywords = utils.get_keywords_dict(movie_pd)
    assert keywords["title"] == 1 and keywords["test"] == 2 and keywords["test1"] == 1
