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
            [213.51862975,  57.92756234,  13.40863054])))
        assert np.all(np.isclose(oe.x_op_err.values, np.array(
            [40.63426603,  2.29118537,  2.6609677])))
        assert np.all(
            np.isclose(
                oe.y_op.values,
                np.array([
       3.17145649e-06, 1.10062999e-05, 3.65341959e-05, 1.15993429e-04,
       3.52243388e-04, 1.02312321e-03, 2.84242260e-03, 7.55309564e-03,
       1.91971583e-02, 4.66685743e-02, 1.08514493e-01, 2.41338486e-01,
       5.13382485e-01, 1.04455458e+00, 2.03281029e+00, 3.78388728e+00,
       6.73682327e+00, 1.14722273e+01, 1.86859835e+01, 2.91111794e+01,
       4.33789731e+01, 6.18264567e+01, 8.42839875e+01, 1.09898412e+02,
       1.37060838e+02, 1.63497447e+02, 1.86545249e+02, 2.03579028e+02,
       2.12499301e+02, 2.12157114e+02, 2.02597144e+02, 1.85048114e+02,
       1.61663372e+02, 1.35087208e+02, 1.07967351e+02, 8.25365465e+01,
       6.03497901e+01, 4.22066478e+01, 2.82332960e+01, 1.80641665e+01,
       1.10547748e+01, 6.47079287e+00, 3.62276965e+00, 1.93999035e+00,
       9.93651387e-01, 4.86792795e-01, 2.28102406e-01, 1.02233026e-01,
       4.38256322e-02, 1.79696985e-02
       ])
            )
        )
        assert np.isclose(oe.dgf, (2.6733916350055233))
        assert np.isclose(oe.trueLinearity,  1.1025635773075495)
        assert np.all(
            np.isclose(
                oe.linearity, np.array([
                  0.04508784837582387, 
                  0.02804107536933482, 
                  0.0191628880252254
                  ])
            )
        )
        assert np.all(
          np.isclose(
            oe.chi2Results['chi2value'] , 
            np.array([67.748733, 41.148962, 2.908089, 0.176378])
            )
          )
        
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
            [242.23058144,  58.80698325])))
        assert np.all(np.isclose(oe.x_op_err.values, np.array(
            [38.45975929,  2.11433569])))
        assert np.all(
            np.isclose(
                oe.y_op.values,
                np.array([8.59387481e-05, 2.35930374e-04, 6.25082165e-04, 1.59826496e-03,
       3.94383617e-03, 9.39176836e-03, 2.15841176e-02, 4.78717872e-02,
       1.02466854e-01, 2.11663310e-01, 4.21955044e-01, 8.11792809e-01,
       1.50724078e+00, 2.70071334e+00, 4.67017078e+00, 7.79373174e+00,
       1.25521050e+01, 1.95094978e+01, 2.92640235e+01, 4.23623849e+01,
       5.91813932e+01, 7.97899855e+01, 1.03817366e+02, 1.30361709e+02,
       1.57975050e+02, 1.84750395e+02, 2.08516617e+02, 2.27119478e+02,
       2.38740727e+02, 2.42190476e+02, 2.37107902e+02, 2.24023415e+02,
       2.04267474e+02, 1.79747741e+02, 1.52646230e+02, 1.25102832e+02,
       9.89479086e+01, 7.55273967e+01, 5.56366318e+01, 3.95526479e+01,
       2.71361802e+01, 1.79671951e+01, 1.14807503e+01, 7.07976235e+00,
       4.21333028e+00, 2.41986283e+00, 1.34126444e+00, 7.17458014e-01,
       3.70371013e-01, 1.84516797e-01])
            )
        )
        assert np.isclose(oe.dgf, 1.9612330067312422)
        assert np.all(
          np.isclose(
            oe.chi2Results['chi2value'].values , 
            np.array([47.36895951, 32.34974831,  0.21562209,  0.07752859])
            )
          )
        assert np.isclose(oe.trueLinearity,  0.04230584990874298)
        assert np.all(
            np.isclose(
                oe.linearity, np.array([0.029214156840052827, 0.0065951044149309464])
            )
        )
