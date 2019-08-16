#!/bin/env python
# -*- coding: utf-8 -*-


import argparse
import sys
from operator import itemgetter

from recsys.common import utils
from recsys.dataset.dataset import MovieLenDataset

parser = argparse.ArgumentParser(description="get zip code list from dataset movie len")
parser.add_argument("--dir", type=str, help="directory of movie len")

if __name__ == "__main__":
    args = parser.parse_args()

    ml = MovieLenDataset(args.dir)
    zip_codes = utils.get_zip_codes_dict(ml.users)


    for item in sorted(zip_codes.items(), key=itemgetter(1), reverse=True):
        print(item[0])
