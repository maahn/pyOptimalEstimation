# -*- coding: utf-8 -*-
"""
pyOptimalEstimation minimal working example

Retrieve N0 and lambda of the drop size distribution N(D) = N0 * exp(-lambda*D)
given a refletivity measurement and prior knowledge about N0 and lambda.
Rayleigh scattering is assumed.

# Copyright (C) 2014-21 Maximilian Maahn, Leipzig University
# maximilian.maahn@uni-leipzig.de
# https://github.com/maahn/pyOptimalEstimation
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import pyOptimalEstimation as pyOE

# Define the forward operator, accepts state vector X [N0,lam] as input, and
# returns measurement vector Y [Ze]


def forward(X, D=np.logspace(-4, -2, 50)):
    if len(X) == 2:
        N0log, lam = X
        mu = 1
    else:
        N0log, lam, mu = X
    N0 = 10**N0log
    dD = np.gradient(D)
    N = N0 * np.exp(-lam * D**mu)
    Z = 1e18 * np.sum(N*D**6*dD)
    return [10*np.log10(Z)]


# define names for X and Y
x_vars = ["N0log", "lam"]
y_vars = ["Ze"]

# prior knowledge. Note that the provided numbers do not have any scientific
# meaning.

# first guess for X
x_ap = [3, 4100]
# covariance matrix for X
x_cov = np.array([[1, 0], [0, 10]])
# covariance matrix for Y
y_cov = np.array([[1]])

# measured observation of Y
y_obs = np.array([10])

# additional data to forward function
forwardKwArgs = {"D": np.logspace(-4, -2, 50)}

# create optimal estimation object
oe = pyOE.optimalEstimation(
    x_vars, x_ap, x_cov, y_vars, y_obs, y_cov, forward,
    forwardKwArgs=forwardKwArgs
    )

# run the retrieval
oe.doRetrieval(maxIter=10, maxTime=10000000.0)

# plot the result
oe.plotIterations()
plt.savefig("oe_result.png")
