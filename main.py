# ======================================================================= 
# This file is part of the CS519_Project_6 project.
#
# Author: Omid Jafari - omidjafari.com
# Copyright (c) 2018
#
# For the full copyright and license information, please view the LICENSE
# file that was distributed with this source code.
# =======================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
import algorithms


for dataset in ["iris", "digits"]:

    # ## Preprocessing ##
    # Iris dataset
    if dataset == "iris":
        print("\n\n************")
        print("Iris dataset")
        print("************")

        # Reading the file
        df = pd.read_csv("iris.data", header=None)

        # Separating x and y
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        names = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
        y = np.array([names[y[i]] for i in range(len(y))])

    # Digits dataset
    elif dataset == "digits":
        print("\n\n************")
        print("Digits dataset")
        print("************")

        # Loading the dataset
        digits = datasets.load_digits()
        x = digits.data
        y = digits.target

    # Standardizing data
    sc_x = StandardScaler()
    x_std = sc_x.fit_transform(x)

    # Splitting data
    x_std_tr, x_std_ts, y_tr, y_ts = train_test_split(x_std, y, test_size=0.3, random_state=1)

    algorithm = algorithms.Algorithms(n_components=2, kernel="rbf", gamma=15, c=100, seed=1, x_train=x_std_tr,
                                      y_train=y_tr, x_test=x_std_ts)

    # ## Regular Logistic Regression ##
    orig_logreg_y_ts_pred = algorithm.run_logisticreg()
    accuracy = np.sum(orig_logreg_y_ts_pred == y_ts) / len(y_ts)
    print("LogReg accuracy before dim reduction: " + str(accuracy))

    for name in ["PCA", "LDA", "Kernel_PCA"]:

        print("\n\n" + name)
        # Reducing the dimensions
        x_tr_reduc, x_ts_reduc = algorithm.call("run_" + name.lower())

        # Logistic Regression on reduced dimensions
        algorithm.x_tr, algorithm.x_ts = x_tr_reduc, x_ts_reduc
        logreg_y_ts_pred = algorithm.run_logisticreg()
        accuracy = np.sum(logreg_y_ts_pred == y_ts) / len(y_ts)
        print("LogReg accuracy after " + name + ": " + str(accuracy))






