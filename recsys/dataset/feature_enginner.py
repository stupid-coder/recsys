#!/bin/env python
# -*- coding: utf-8

import re
from datetime import datetime


def age(value):
    value = int(value)
    if value == 1:
        return 0
    elif value == 18:
        return 1
    elif value == 25:
        return 2
    elif value == 35:
        return 3
    elif value == 45:
        return 5
    elif value == 50:
        return 6
    elif value == 56:
        return 7
    else:
        return -1

def gender(value):
    if value == "M":
        return 0
    elif value == "F":
        return 1
    else:
        return -1

def genres(value):
    _genres = { v:i for i,v in enumerate(["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"])}

    def _get_idx(v):
        if v in _genres:
            return _genres[v]
        else:
            return -1

    return "|".join([str(_get_idx(v)) for v in value.split("|")])

def timestamp(value):
    return datetime.fromtimestamp(int(value))

def title(keywords):
    _keywords = {}
    with open(keywords, 'r') as f:
        lnum = 0
        for line in f:
            _keywords[line.strip()] = lnum
            lnum += 1

    def _get_idx(v):
        keywords, _ = split_title(v)
        return "|".join([str(_keywords[k.lower()]) for k in keywords if k != ""])

    return _get_idx

def year(value):
    return split_title(value)[1]

def split_title(value):
    keywords = re.split(r"[, \(\):\!&$'\?\.\-#;]", value)
    return keywords[:-3], keywords[-2]

def zip_code(zip_codes):
    _zip_codes = {}
    with open(zip_codes, 'r') as f:
        lnum = 0
        for line in f:
            _zip_codes[line.strip()] = lnum
            lnum += 1

    def _get_idx(zip_code):
        zip_code = zip_code.split("-")[0]
        return _zip_codes[zip_code]

    return _get_idx
