# -*- coding: utf-8 -*-

import numpy as np
import pandas as pn
import scipy
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
        y_obs = forward_simple(x_truth) + \
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
        )
        oe.doRetrieval()
        chi2passed, chi2value, chi2critical = oe.chiSquareTest()
        linearity, trueLinearityChi2, trueLinearityChi2Critical = oe.linearityTest()

        print('np', np.__version__)
        print('pn', pn.__version__)
        print('pyOE', pyOE.__version__)
        print('scipy', scipy.__version__)

        assert np.all(np.isclose(oe.x_op.values, np.array(
            [230.75639479,  58.49351178,  12.32118448])))
        assert np.all(np.isclose(oe.x_op_err.values, np.array(
            [42.940902,  2.05667214,  2.4442318])))
        assert np.all(
            np.isclose(
                oe.y_op.values,
                np.array(
                    [8.07132255e-08, 3.57601384e-07,
                     1.50303019e-06, 5.99308237e-06,
                     2.26697545e-05, 8.13499728e-05,
                     2.76937671e-04, 8.94377132e-04,
                     2.74014386e-03, 7.96416170e-03,
                     2.19594168e-02, 5.74401494e-02,
                     1.42535928e-01, 3.35542210e-01,
                     7.49348775e-01, 1.58757726e+00,
                     3.19080133e+00, 6.08385266e+00,
                     1.10045335e+01, 1.88833313e+01,
                     3.07396995e+01, 4.74716852e+01,
                     6.95478498e+01, 9.66600005e+01,
                     1.27445324e+02, 1.59409810e+02,
                     1.89156040e+02, 2.12931252e+02,
                     2.27390663e+02, 2.30366793e+02,
                     2.21401803e+02, 2.01862876e+02,
                     1.74600622e+02, 1.43267982e+02,
                     1.11523535e+02, 8.23565110e+01,
                     5.76956909e+01, 3.83444793e+01,
                     2.41755486e+01, 1.44598524e+01,
                     8.20475101e+00, 4.41652793e+00,
                     2.25533253e+00, 1.09258243e+00,
                     5.02125028e-01, 2.18919049e-01,
                     9.05459987e-02, 3.55278561e-02,
                     1.32246067e-02, 4.66993202e-03]
                )
            )
        )
        assert np.isclose(oe.dgf, (2.7132392503933556))
        assert np.isclose(oe.trueLinearity,  0.41529831393972894)
        assert np.all(np.array(oe.linearity) < 1)
        assert np.all(chi2passed)

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
        chi2passed, chi2value, chi2critical = oe.chiSquareTest()
        linearity, trueLinearityChi2, trueLinearityChi2Critical = oe.linearityTest()

        print('np', np.__version__)
        print('pn', pn.__version__)
        print('pyOE', pyOE.__version__)
        print('scipy', scipy.__version__)

        assert np.all(np.isclose(oe.x_op.values, np.array(
            [255.30992222,  58.68130862])))
        assert np.all(np.isclose(oe.x_op_err.values, np.array(
            [38.59979542,  2.0071929])))
        assert np.all(
            np.isclose(
                oe.y_op.values,
                np.array([9.66145500e-05, 2.64647052e-04,
                          6.99600329e-04, 1.78480744e-03,
                          4.39431462e-03, 1.04411743e-02,
                          2.39423053e-02, 5.29835436e-02,
                          1.13155184e-01, 2.33220288e-01,
                          4.63891718e-01, 8.90482377e-01,
                          1.64965245e+00, 2.94929357e+00,
                          5.08864283e+00, 8.47313942e+00,
                          1.36158624e+01, 2.11156461e+01,
                          3.16025417e+01, 4.56455107e+01,
                          6.36256959e+01, 8.55904756e+01,
                          1.11116039e+02, 1.39215145e+02,
                          1.68327329e+02, 1.96417962e+02,
                          2.21190355e+02, 2.40386232e+02,
                          2.52122389e+02, 2.55194702e+02,
                          2.49281652e+02, 2.34999746e+02,
                          2.13797631e+02, 1.87714060e+02,
                          1.59055663e+02, 1.30064833e+02,
                          1.02642934e+02, 7.81729764e+01,
                          5.74569615e+01, 4.07555802e+01,
                          2.78990827e+01, 1.84310971e+01,
                          1.17508929e+01, 7.23017763e+00,
                          4.29324314e+00, 2.46025669e+00,
                          1.36061037e+00, 7.26182114e-01,
                          3.74038011e-01, 1.85927808e-01])
            )
        )
        assert np.isclose(oe.dgf, 1.9611398655015124)
        assert np.isclose(oe.trueLinearity,  0.039634853402863594)
        assert np.all(np.array(oe.linearity) < 1)
        assert np.all(chi2passed)
