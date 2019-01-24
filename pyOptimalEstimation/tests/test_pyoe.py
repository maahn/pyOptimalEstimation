# -*- coding: utf-8 -*-

import numpy as np
import pandas as pn
from copy import deepcopy
import pyOptimalEstimation as pyOE


def forward_simple(X):
    z = np.linspace(1, 99, 50)
    N = X['N']
    R = X['R']
    W = X['W']
    F = N * np.exp(-((z-R)/W)**2)
    return F


class TestFullRetrieval(object):

    def test_simple(self):
        x_vars = ['N', 'R', 'W']
        x_truth = pn.Series([300., 60., 10.], index=x_vars)
        x_a = pn.Series([200., 50., 15.], index=x_vars)
        x_cov = pn.DataFrame([[200.**2,      0.,     0.],
                              [0.,  50.**2,     0.],
                              [0.,      0.,  5.**2]],
                             index=x_vars,
                             columns=x_vars,
                             )
        y_vars = ['z%02i' % i for i in range(50)]
        y_cov = pn.DataFrame(np.identity(50) * 100**2,
                             index=y_vars,
                             columns=y_vars)
        np.random.seed(1)
        y_obs = forward_simple(x_truth) + np.random.normal(loc=0, scale=100, size=50)
        oe = pyOE.optimalEstimation(
            x_vars,
            x_a,
            x_cov,
            y_vars,
            y_obs,
            y_cov,
            forward_simple,
            forwardKwArgs={},
            x_truth=x_truth,
        )
        oe.doRetrieval()
        oe.chiSquareTest()
        oe.linearityTest()

        assert np.all(np.isclose(oe.x_op.values, np.array(
            [216.67732376,  58.16498463,  12.62412649])))
        assert np.all(np.isclose(oe.x_op_err.values, np.array(
            [40.71776776,  2.07152667,  2.51450318])))
        assert np.all(
            np.isclose(
                oe.y_op.values,
                np.array([2.69557604e-07, 1.10375135e-06,
                          4.29823492e-06, 1.59187253e-05,
                          5.60693694e-05, 1.87820198e-04,
                          5.98353942e-04, 1.81289754e-03,
                          5.22381278e-03, 1.43153242e-02,
                          3.73090348e-02, 9.24753702e-02,
                          2.17990408e-01, 4.88706293e-01,
                          1.04197601e+00, 2.11284066e+00,
                          4.07450649e+00, 7.47278561e+00,
                          1.30343468e+01, 2.16219701e+01,
                          3.41114745e+01, 5.11805517e+01,
                          7.30312399e+01, 9.91086650e+01,
                          1.27912745e+02, 1.57005639e+02,
                          1.83280358e+02, 2.03477240e+02,
                          2.14839926e+02, 2.15731413e+02,
                          2.06020779e+02, 1.87114697e+02,
                          1.61623318e+02, 1.32769822e+02,
                          1.03727515e+02, 7.70704350e+01,
                          5.44604078e+01, 3.65993362e+01,
                          2.33918607e+01, 1.42185585e+01,
                          8.21950397e+00, 4.51892248e+00,
                          2.36278069e+00, 1.17492779e+00,
                          5.55645982e-01, 2.49910453e-01,
                          1.06898080e-01, 4.34865176e-02,
                          1.68243617e-02, 6.19044403e-03])
            )
        )
        assert np.isclose(oe.dgf, 2.7039260453696317)
        assert np.isclose(oe.trueNonlinearity,  0.5892181071502255)
        assert np.all(
            np.isclose(
                oe.nonlinearity, np.array([0.0172169, 0.01304629, 0.01061978])
            )
        )
        assert np.isclose(oe.chi2, 64.21108547675885)

    def test_simple_withB(self):
        x_vars = ['N', 'R']
        b_vars = ['W']

        x_truth = pn.Series([300., 60.], index=x_vars)
        x_a = pn.Series([200., 50.], index=x_vars)
        x_cov = pn.DataFrame([[200.**2,      0.],
                              [0.,  50.**2]],
                             index=x_vars,
                             columns=x_vars,
                             )

        b_vars = ['W']
        b_param = pn.Series(15, index=b_vars)
        b_cov = pn.DataFrame([[5**2]], index=b_vars, columns=b_vars)

        y_vars = ['z%02i' % i for i in range(50)]
        y_cov = pn.DataFrame(np.identity(50) * 100**2,
                             index=y_vars,
                             columns=y_vars)
        np.random.seed(1)
        y_obs = forward_simple(pn.concat((x_truth, b_param))) + \
            np.random.normal(loc=0, scale=100, size=50)

        oe = pyOE.optimalEstimation(
            x_vars,
            x_a,
            x_cov,
            y_vars,
            y_obs,
            y_cov,
            forward_simple,
            forwardKwArgs={},
            x_truth=x_truth,
            b_vars=b_vars,
            b_p=b_param,
            S_b=b_cov,
        )
        oe.doRetrieval()
        oe.chiSquareTest()
        oe.linearityTest()

        assert np.all(np.isclose(oe.x_op.values, np.array(
            [252.7093795,  58.47319368])))
        assert np.all(np.isclose(oe.x_op_err.values, np.array(
            [37.6333103,  1.93583752])))
        assert np.all(
            np.isclose(
                oe.y_op.values,
                np.array([1.06378560e-04, 2.90316569e-04,
                          7.64624051e-04, 1.94349087e-03,
                          4.76733284e-03, 1.12856574e-02,
                          2.57831911e-02, 5.68466519e-02,
                          1.20957147e-01, 2.48379984e-01,
                          4.92220927e-01, 9.41373507e-01,
                          1.73748990e+00, 3.09486011e+00,
                          5.32008016e+00, 8.82579321e+00,
                          1.41301814e+01, 2.18323330e+01,
                          3.25544958e+01, 4.68468310e+01,
                          6.50590746e+01, 8.71954782e+01,
                          1.12781655e+02, 1.40780137e+02,
                          1.69590969e+02, 1.97161665e+02,
                          2.21207909e+02, 2.39517496e+02,
                          2.50283520e+02, 2.52397869e+02,
                          2.45639119e+02, 2.30710732e+02,
                          2.09120439e+02, 1.82929432e+02,
                          1.54429088e+02, 1.25815181e+02,
                          9.89225733e+01, 7.50613173e+01,
                          5.49661574e+01, 3.88448091e+01,
                          2.64928709e+01, 1.74374700e+01,
                          1.10763416e+01, 6.78996647e+00,
                          4.01695841e+00, 2.29342984e+00,
                          1.26366513e+00, 6.71950008e-01,
                          3.44826269e-01, 1.70774153e-01])
            )
        )
        assert np.isclose(oe.dgf, 1.9630943621484518)
        assert np.isclose(oe.chi2, 60.41671208566697)
        assert np.isclose(oe.trueNonlinearity,  0.02364625922873849)
        assert np.all(
            np.isclose(
                oe.nonlinearity, np.array([1.93185672e-05, 6.04938579e-03])
            )
        )
