#!/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict

from recsys.dataset import feature_enginner


def get_keywords_dict(movie_len):
    keywords = defaultdict(int)
    for i in range(len(movie_len)):
        title = movie_len.iloc[i]["title"]
        words, _ = feature_enginner.split_title(title)
        for w in words:
            if w == "":
                continue
            w = w.lower()
            keywords[w] = keywords[w] + 1
    return keywords



def get_zip_codes_dict(movie_len):
    zip_codes = defaultdict(int)
    for i in range(len(movie_len)):
        zip_code = movie_len.iloc[i]["zip-code"].split("-")[0]
        zip_codes[zip_code] = zip_codes[zip_code] + 1
    return zip_codes
