#！/bin/env python
# -*- coding: utf-8 -*-


import os

import numpy as np
import pandas as pd

from recsys.common.cache import cached
from recsys.dataset import feature_enginner


class Dataset(object):
    def __init__(self, name, path):
        self._name = name
        self._path = path

    def code(self):
        raise NotImplementedError("[{}] not implemented code".format(self.__class__))

    def load(self):
        raise NotImplementedError("[{}] not implemented load".format(self.__class__))

    def R(self):
        raise NotImplementedError("[{}] not implemented rating".format(self.__class__))


class MovieLenDataset(Dataset):
    """
    ml-1m 数据集，包含6040用户和3952电影
    """

    def _code_users(self):
        self._users_code = pd.concat(
            [self.users["uid"] - 1,
             self.users["gender"].apply(feature_enginner.gender),
             self.users["age"].apply(feature_enginner.age),
             self.users["occupation"],
             self.users["zip-code"].apply(feature_enginner.zip_code("recsys/dataset/zip_code"))],
            axis=1)

    def _code_movies(self):
        self._movies_code = pd.concat(
            [self.movies["mid"] - 1,
             self.movies["title"].apply(feature_enginner.title("recsys/dataset/keywords")),
             self.movies["genres"].apply(feature_enginner.genres)],
            axis=1)
        self._movies_code["year"] = self.movies["title"].apply(feature_enginner.year)

    def _code_ratings(self):
        self._ratings_code = pd.concat(
            [self.ratings["uid"]-1,
             self.ratings["mid"]-1,
             self.ratings["rating"]],
            axis=1)


    def _read(self):
        fname = os.path.join(self._path, "users.dat")
        self._users = pd.read_csv(fname,
                                  sep="::",
                                  header=None,
                                  names=["uid","gender","age","occupation","zip-code"],
                                  engine="python")

        fname = os.path.join(self._path, "movies.dat")
        self._movies = pd.read_csv(fname,
                                   sep="::",
                                   names=["mid", "title", "genres"],
                                   engine="python",
                                   encoding="latin_1")

        fname = os.path.join(self._path, "ratings.dat")
        self._ratings = pd.read_csv(fname,
                                    sep="::",
                                    names=["uid", "mid", "rating", "timestamp"],
                                    parse_dates=[3],
                                    date_parser=feature_enginner.timestamp,
                                    engine="python")
        print("[MovieLenDataset] load data finished")
        self._code_users()
        self._code_movies()
        self._code_ratings()
        print("[MovieLenDataset] code data finished")

    def __init__(self, path):
        super().__init__("movie-len", path)
        self._read()

    @property
    def users(self):
        return self._users

    @users.setter
    def users(self, users):
        self._users = users

    @property
    def users_code(self):
        return self._users_code

    @property
    def movies(self):
        return self._movies

    @movies.setter
    def movies(self, movies):
        self._movies = movies

    @property
    def movies_code(self):
        return self._movies_code

    @property
    def ratings(self):
        return self._ratings

    @ratings.setter
    def ratings(self, ratings):
        self._ratings = ratings

    @property
    def ratings_code(self):
        return self._ratings_code

    @property
    @cached("recsys/dataset/cache/cache.npy")
    def R(self):
        if "_R" not in self.__dict__:
            fname = os.path.join(self._path, "rating.npy")
            if os.path.exists(fname):
                self._R = np.load(open(fname, "rb"))
            else:
                self._R = np.zeros((6040, 3952), dtype=int)
                for i in range(len(self.ratings_code)):
                    uid, mid, rating = self.ratings_code.iloc[i]
                    self._R[uid, mid] = rating
                np.save(open(fname, "wb"), self._R)
            print("[MovieLenDataset] load R finished")
        return self._R
