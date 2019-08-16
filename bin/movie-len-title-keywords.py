#!/bin/env python
# -*- coding: utf-8 -*-
import argparse
import sys
from operator import itemgetter

from recsys.common import utils
from recsys.dataset.dataset import MovieLenDataset

parser = argparse.ArgumentParser(description="get keywords list from datset movie len")
parser.add_argument("--dir", type=str, help="directory of movie len")


if __name__ == "__main__":
    args = parser.parse_args()

    ml = MovieLenDataset(args.dir)
    keywords = utils.get_keywords_dict(ml.movies)

    for item in sorted(keywords.items(), key=itemgetter(1), reverse=True):
        print(item[0])
