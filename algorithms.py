# ======================================================================= 
# This file is part of the CS519_Project_6 project.
#
# Author: Omid Jafari - omidjafari.com
# Copyright (c) 2018
#
# For the full copyright and license information, please view the LICENSE
# file that was distributed with this source code.
# =======================================================================

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from time import time
import inspect


class Algorithms(object):
    def __init__(self, n_components, kernel, gamma, c, seed, x_train=[], y_train=[], x_test=[]):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.c = c
        self.seed = seed
        self.x_tr = x_train
        self.y_tr = y_train
        self.x_ts = x_test
        self.__obj = None

    def call(self, method):
        return getattr(self, method)()

    def __fit(self):
        start = int(round(time() * 1000))
        self.__obj.fit(self.x_tr, self.y_tr)
        end = int(round(time() * 1000)) - start
        print(inspect.stack()[1][3].split("_", 1)[1] + f" training time: {end if end > 0 else 0} ms")

    def __predict(self):
        start = int(round(time() * 1000))
        y_pred = self.__obj.predict(self.x_ts)
        end = int(round(time() * 1000)) - start
        print(inspect.stack()[1][3].split("_", 1)[1] + f" prediction time: {end if end > 0 else 0} ms")
        return y_pred

    def __transform(self):
        start = int(round(time() * 1000))
        x_test = self.__obj.transform(self.x_ts)
        end = int(round(time() * 1000)) - start
        print(inspect.stack()[1][3].split("_", 1)[1] + f" transform time: {end if end > 0 else 0} ms")
        return x_test

    def __fit_transform(self):
        start = int(round(time() * 1000))
        x_train_reduced = self.__obj.fit_transform(self.x_tr, y=self.y_tr)
        end = int(round(time() * 1000)) - start
        print(inspect.stack()[1][3].split("_", 1)[1] + f" fit_transform time: {end if end > 0 else 0} ms")
        return x_train_reduced

    def run_pca(self):
        self.__obj = PCA(n_components=self.n_components)
        x_train_pca = self.__fit_transform()
        x_test_pca = self.__transform()
        return x_train_pca, x_test_pca

    def run_lda(self):
        self.__obj = LinearDiscriminantAnalysis(n_components=self.n_components)
        x_train_lda = self.__fit_transform()
        x_test_lda = self.__transform()
        return x_train_lda, x_test_lda

    def run_kernel_pca(self):
        self.__obj = KernelPCA(n_components=self.n_components, kernel=self.kernel, gamma=self.gamma)
        x_train_kpca = self.__fit_transform()
        x_test_kpca = self.__transform()
        return x_train_kpca, x_test_kpca

    def run_logisticreg(self):
        self.__obj = LogisticRegression(random_state=self.seed, C=self.c)
        self.__fit()
        return self.__predict()


















