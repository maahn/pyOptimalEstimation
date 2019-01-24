# -*- coding: utf-8 -*-
# pyOptimalEstimation

# Copyright (C) 2014-15 Maximilian Maahn, IGMK (mmaahn_(AT)_meteo.uni-koeln.de)
# http://gop.meteo.uni-koeln.de/software


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import time
from copy import deepcopy
import warnings

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.ticker as ticker
import pandas as pd


class optimalEstimation(object):
    r'''
    The core optimalEstimation class, which contains all required parameters.

    Parameters
    ----------
    x_vars : list of str
        names of the elements of state vector x.
    x_a : pd.Series or list or np.ndarray
        prior information of state x.
    S_a : pd.DataFrame or list or np.ndarray
        covariance matrix of state x.
    y_vars : list of str
        names of the elements of state vector x
    y_obs : pd.Series or list or np.ndarray
        observed measurement vector y.
    S_y : pd.DataFrame or list or np.ndarray
        covariance matrix of measurement y. If there is no b vector, S_y
        is sequal to S_e
    forward : function
        forward model expected as ``forward(xb,**forwardKwArgs): return y``
        with xb = pd.concat((x,b)).
    x_truth : pd.Series or list or np.ndarray, optional
        If truth of state x is known, it can added to the data object. If
        provided, the value will be used for the routines testLinearity and
        plotIterations, but _not_ by the retrieval itself. Defaults to None/
    b_vars : list of str, optional
        names of the elements of parameter vector b. defaults to [].
    b_p : pd.Series or list or np.ndarray.
        parameter vector b.  defaults to []. Note that defining b_p makes
        only sence if S_b != 0. Otherwise it is easier (and cheaper) to
        hardcode b into the forward operator.
    S_b : pd.DataFrame or list or np.ndarray
        covariance matrix of parameter b. defaults to [[]].
    forwardKwArgs : dict,optional
        additional keyword arguments for forward function.
    x_lowerLimit : dict, optional
        reset state vector x[key] to x_lowerLimit[key] in case x_lowerLimit is
        undercut. defaults to {}.
    x_upperLimitn : dict, optional
        reset state vector x[key] to x_upperLimit[key] in case x_upperLimit is
        exceeded. defaults to {}.
    disturbance : float or dict of floats, optional
        relative disturbance of statet vector x to estimate the Jacobian. Can
        be specified for every element of x seperately. Defaults to 1.01.
    useFactorInJac : bool,optional
        True if disturbance should be applied by multiplication, False if it
        should by applied by additiion. Defaults to True.
    gammaFactor : list of floats, optional
        Use additional gamma parameter for retrieval, see [1]_.

    Attributes
    ----------
    converged : boolean
      True if retriveal converged successfully
    convI : int
      iteration where convergence was achieved
    K_i : list of pd.DataFrame
      list of Jacobians for iteration i.
    x_i : list of pd.Series
      iterations of state vector x
    y_i : list of pd.Series
      iterations of measurement vector y
    dgf_i : list of float
      degrees of freedom for each iteration
    A_i  : list of pd.DataFrame
      Averaging kernel for each iteration
    d_i2 : list of float
      convergence criteria for each iteration
    S_aposterior_i : list of pd.DataFrame
      a posteriori covariance matrix of x for each iteration
    gam_i : list of floats
      gamma parameters used in retrievals, see also `gammaFactor` and  [1]_.
    x_op : pd.Series
      optimal state given the observations, i.e. retrieval solution
    y_op : pd.Series
      Optimal y, i.e. observation associated with retrieval solution
    S_op : pd.DataFrame
      covariance of x_op, i.e. solution uncertainty
    x_op_err : pd.Series
      1 sigma errors of x_op. derived with sqrt(diag(S_op))
    dgf : float
      total degrees of freedom for signal of the retrieval solution
    dgf_x : pd.Series
      degrees of freedom for signal per state variable


    Returns
    -------

    pyOptimalEstimation object
      returns the pyOptimalEstimation object


    References
    ----------
    .. [1] Turner, D. D., and U. Löhnert, 2014: Information Content and
    Uncertainties in Thermodynamic Profiles and Liquid Cloud Properties
    Retrieved from the Ground-Based Atmospheric Emitted Radiance
    Interferometer (AERI). Journal of Applied Meteorology & Climatology, 53,
    752–771, doi:10.1175/JAMC-D-13-0126.1.

    '''

    def __init__(self,
                 x_vars,
                 x_a,
                 S_a,
                 y_vars,
                 y_obs,
                 S_y,
                 forward,
                 x_truth=None,
                 b_vars=[],
                 b_p=[],
                 S_b=[[]],
                 x_lowerLimit={},
                 x_upperLimit={},
                 useFactorInJac=True,
                 gammaFactor=None,
                 disturbance=1.01,
                 convergenceFactor=10,
                 forwardKwArgs={},
                 ):

        for inVar in [x_a, S_a, S_y, y_obs]:
            assert not np.any(np.isnan(inVar))

        self.x_vars = x_vars
        self.x_a = pd.Series(x_a, index=self.x_vars)
        self.S_a = pd.DataFrame(
            S_a, index=self.x_vars, columns=self.x_vars)
        self.x_n = len(self.x_vars)
        self.y_vars = y_vars
        self.S_y = pd.DataFrame(
            S_y, index=self.y_vars, columns=self.y_vars)
        self.y_obs = pd.Series(y_obs, index=self.y_vars)
        self.y_n = len(self.y_vars)
        self.forward = forward
        self.x_truth = pd.Series(x_truth, index=self.x_vars)
        try:
            # We want to save at least the name because the forward function
            # is removed for saving
            self.forward_name = forward.__name__
        except AttributeError:
            self.forward_name = None
        self.b_vars = b_vars
        self.b_n = len(self.b_vars)
        self.b_p = pd.Series(b_p, index=self.b_vars)
        self.S_b = pd.DataFrame(
            S_b, index=self.b_vars, columns=self.b_vars)

        self.forwardKwArgs = forwardKwArgs
        self.x_lowerLimit = x_lowerLimit
        self.x_upperLimit = x_upperLimit
        self.useFactorInJac = useFactorInJac
        self.gammaFactor = gammaFactor
        self.disturbance = disturbance
        self.convergenceFactor = convergenceFactor

        self.converged = False
        self.K_i = None
        self.x_i = None
        self.y_i = None
        self.dgf_i = None
        self.A_i = None
        self.d_i2 = None
        self.S_aposterior_i = None
        self.gam_i = None
        self.convI = None
        self.x_op = None
        self.y_op = None
        self.S_op = None
        self.x_op_err = None
        self.dgf = None
        self.dgf_x = None

    def getJacobian(self, xb):
        r'''
        estimate Jacobian using the forward model and the specified disturbance

        Parameters
        ----------
        xb  : pd.Series or list or np.ndarray
          combination of state vector x and parameter vector b

        Returns
        -------
        pd.DataFrame
          Jacobian around x
        pd.DataFrame
          Jacobian around b
        '''
        xb_vars = self.x_vars + self.b_vars
        xb = pd.Series(xb, index=xb_vars, dtype=float)

        # If a factor is used to disturb xb, xb must not be zero.
        assert not (self.useFactorInJac and np.any(xb == 0))

        if type(self.disturbance) == float:
            disturbances = dict()
            for key in xb_vars:
                disturbances[key] = self.disturbance
        elif type(self.disturbance) == dict:
            disturbances = self.disturbance
        else:
            raise TypeError("disturbance must be type dict or float")

        disturbedKeys = ["reference"]
        for tup in xb_vars:
            disturbedKeys.append("disturbed %s" % tup)
        self.xb_disturbed = pd.DataFrame(
            columns=xb_vars, index=disturbedKeys, dtype=float)
        self.xb_disturbed.loc["reference"] = xb
        for xb_key in xb_vars:
            disturbed_xb_key = "disturbed %s" % xb_key
            self.xb_disturbed.loc[disturbed_xb_key] = xb
            # apply disturbance here!!
            if self.useFactorInJac:
                self.xb_disturbed[xb_key][disturbed_xb_key] = xb[xb_key] * \
                    disturbances[xb_key]
            else:
                self.xb_disturbed[xb_key][disturbed_xb_key] = xb[xb_key] + \
                    disturbances[xb_key]
        self.y_disturbed = pd.DataFrame(
            columns=self.y_vars,
            index=disturbedKeys,
            dtype=np.float64
        )
        for xb_dist in self.xb_disturbed.index:
            self.y_disturbed.loc[xb_dist] = self.forward(
                self.xb_disturbed.loc[xb_dist], **self.forwardKwArgs)

        y = self.y_disturbed.loc["reference"]

        # remove the reference from the disturbed keys!
        disturbedKeys = disturbedKeys[1:]
        # create an empty jacobian matrix
        jacobian = pd.DataFrame(np.ones(
            (self.y_n, self.x_n+self.b_n)
        ), index=self.y_vars, columns=disturbedKeys)
        # calc Jacobian
        for y_key in self.y_vars:
            for x_key in xb_vars:
                # realtive disturbance
                if self.useFactorInJac:
                    dist = xb[x_key] * (disturbances[x_key] - 1)
                else:
                    dist = disturbances[x_key]
                jacobian["disturbed "+x_key][y_key] = (
                    self.y_disturbed[y_key]["disturbed "+x_key] - y[y_key]
                ) / dist

        jacobian[np.isnan(jacobian) | np.isinf(jacobian)] = 0.
        jacobian_x = jacobian[["disturbed %s" % s for s in self.x_vars]]
        jacobian_b = jacobian[["disturbed %s" % s for s in self.b_vars]]

        return jacobian_x, jacobian_b

    def doRetrieval(self, maxIter=10, x_0=None, maxTime=1e7):
        r"""
        run the retrieval

        Parameters
        ----------
        maxIter  : int, optional
          maximum number of iterations, defaults to 10
        x_0  : pd.Series or list or np.ndarray, optional
          first guess for x. If x_0 == None, x_a is taken as first guess.
        maxTime  : int, optional
          maximum runTime, defaults to 1e7 (~ 4 months).
          Note that the forward model is *not* killed if time is exceeded

        Returns
        -------
        bool
          True is convergence was obtained.

        """

        assert maxIter > 0
        self.converged = False
        startTime = time.time()

        S_a = np.array(self.S_a)  # Covariance of prior estimate of x
        S_a_inv = _invertMatrix(S_a)  # S_a inverted
        self.K_i = [0]*maxIter  # list of jacobians
        self.K_b_i = [0]*maxIter  # list of jacobians for parameter vector
        self.x_i = [0]*(maxIter+1)
        self.y_i = [0]*maxIter
        self.dgf_i = [0]*maxIter
        self.H_i = [0]*maxIter  # Shannon information content
        self.A_i = [0]*maxIter
        self.d_i2 = [0]*maxIter  # convergence criteria
        self.S_aposterior_i = [0] * maxIter
        # self.Pxy_i = [0] *maxIter
        self.gam_i = [1]*maxIter
        if self.gammaFactor:
            assert len(self.gammaFactor) <= maxIter
            self.gam_i[:len(self.gammaFactor)] = self.gammaFactor

        if x_0 is None:
            self.x_i[0] = self.x_a
        else:
            self.x_i[0] = pd.Series(x_0, index=self.x_vars)
        self.d_i2[0] = 1e333

        for i in range(maxIter):

            self.K_i[i], self.K_b_i[i] = self.getJacobian(
                pd.concat((self.x_i[i], self.b_p)))

            if np.sum(self.S_b.shape) > 0:
                S_Ep_b = self.K_b_i[i].values.dot(
                    self.S_b.values).dot(self.K_b_i[i].values.T)
            else:
                S_Ep_b = 0
            # S_Epsilon Covariance of measurement noise including parameter
            # uncertainty (Rodgers, sec 3.4.3)
            S_Ep = self.S_y.values + S_Ep_b
            S_Ep_inv = _invertMatrix(S_Ep)  # S_Ep inverted

            assert np.all(self.y_disturbed.keys() == self.S_y.keys())
            assert np.all(self.S_y.keys() == self.K_i[i].index)
            assert np.all(self.S_a.index == self.x_a.index)
            assert np.all(self.x_a.index.tolist(
            )+self.b_p.index.tolist() == self.xb_disturbed.columns)
            assert np.all(self.xb_disturbed.index[1:].tolist(
            ) == self.K_i[i].columns.tolist()+self.K_b_i[i].columns.tolist())

            self.y_i[i] = self.y_disturbed.loc["reference"]
            K = np.array(self.K_i[i])

            # reformulated using Turner and Löhnert 2013:
            B = (self.gam_i[i] * S_a_inv) + K.T.dot(S_Ep_inv.dot(K))  # eq 3
            B_inv = _invertMatrix(B)
            self.S_aposterior_i[i] = B_inv.dot(
                (self.gam_i[i]**2 * S_a_inv) + K.T.dot(S_Ep_inv.dot(K))
            ).dot(B_inv)  # eq2
            self.S_aposterior_i[i] = pd.DataFrame(
                self.S_aposterior_i[i],
                index=self.x_a.index,
                columns=self.x_a.index
            )
            G = B_inv.dot(K.T.dot(S_Ep_inv))
            self.A_i[i] = G.dot(K)  # eq 4
            self.x_i[i+1] = self.x_a +\
                _invertMatrix((self.gam_i[i] * S_a_inv) +
                              K.T.dot(S_Ep_inv.dot(K))).dot(
                K.T.dot(S_Ep_inv.dot(self.y_obs - self.y_i[i] +
                                     K.dot(self.x_i[i]-self.x_a))))  # eq 1
            self.dgf_i[i] = np.trace(self.A_i[i])
            # eq. 2.80 Rodgers
            self.H_i[i] = -0.5 * \
                np.log(np.linalg.det(np.identity(self.x_n) - self.A_i[i]))

            # check whether i+1 is valid
            for jj, xKey in enumerate(self.x_vars):
                if (xKey in self.x_lowerLimit.keys()) and (
                        self.x_i[i+1][jj] < self.x_lowerLimit[xKey]):
                    print("#"*60)
                    print("reset due to x_lowerLimit: %s from %f to %f in "
                          "iteration %d" % (
                              xKey, self.x_i[i+1][jj], self.x_a[jj], i))
                    self.x_i[i+1][jj] = self.x_a[jj]
                if (xKey in self.x_upperLimit.keys()) and (
                        self.x_i[i+1][jj] > self.x_upperLimit[xKey]):
                    print("#"*60)
                    print("reset due to x_upperLimit: %s from %f to %f in "
                          "iteration %d" % (
                              xKey, self.x_i[i+1][jj], self.x_a[jj], i))
                    self.x_i[i+1][jj] = self.x_a[jj]
                if np.isnan(self.x_i[i+1][jj]):
                    print("#"*60)
                    print("reset due to nan: %s from %f to %f in iteration "
                          "%d" % (
                              xKey, self.x_i[i+1][jj], self.x_a[jj], i))
                    self.x_i[i+1][jj] = self.x_a[jj]

            # convergence criteria  eq 6
            self.d_i2[i] = (self.x_i[i] - self.x_i[i+1]).T.dot(_invertMatrix(
                self.S_aposterior_i[i])).dot(self.x_i[i] - self.x_i[i+1])

            # stop if we converged in the step before
            if self.converged:
                print("%.2f s, iteration %i, degrees of freedom: %.2f of %i. "
                      " Done.  %.3f" % (
                          time.time()-startTime, i, self.dgf_i[i], self.x_n,
                          self.d_i2[i]))
                break

            elif ((time.time()-startTime) > maxTime):
                print("%.2f s, iteration %i, degrees of freedom: %.2f of %i."
                      " maximum Time exceeded! STOP  %.3f" % (
                          time.time()-startTime, i, self.dgf_i[i], self.x_n,
                          self.d_i2[i]))

                self.converged = False

                break

            # calculate the convergence criteria
            if i != 0:
                if (np.abs(self.d_i2[i]) < self.y_n/float(
                    self.convergenceFactor)) and (self.gam_i[i] == 1
                                                  ) and (self.d_i2[i] != 0):
                    print("%.2f s, iteration %i, degrees of freedom: %.2f of"
                          " %i. convergence criteria fullfilled  %.3f" % (
                              time.time() -
                              startTime, i, self.dgf_i[i], self.x_n,
                              self.d_i2[i]))
                    self.converged = True
                elif (i > 1) and (self.dgf_i[i] == 0):
                    print("%.2f s, iteration %i, degrees of freedom: %.2f of "
                          "%i.degrees of freedom 0! STOP  %.3f" % (
                              time.time() -
                              startTime, i, self.dgf_i[i], self.x_n,
                              self.d_i2[i]))
                    self.converged = False

                    break
                else:
                    print("%.2f s, iteration %i, degrees of freedom:"
                          " %.2f of %i. convergence criteria NOT fullfilled "
                          " %.3f" % (
                              time.time()-startTime, i, self.dgf_i[i],
                              self.x_n, self.d_i2[i]))

        self.K_i = self.K_i[:i+1]
        self.K_b_i = self.K_b_i[:i+1]
        self.x_i = self.x_i[:i+2]
        self.y_i = self.y_i[:i+1]
        self.dgf_i = self.dgf_i[:i+1]
        self.A_i = self.A_i[:i+1]
        self.H_i = self.H_i[:i+1]
        self.d_i2 = self.d_i2[:i+1]
        self.S_aposterior_i = self.S_aposterior_i[:i+1]

        self.gam_i = self.gam_i[:i+1]
        if self.converged:
            self.convI = i

            self.x_op = self.x_i[i]
            self.y_op = self.y_i[i]
            self.S_op = self.S_aposterior_i[i]
            self.x_op_err = np.sqrt(
                pd.Series(np.diag(
                    self.S_aposterior_i[self.convI]), index=self.x_vars)
            )
            self.dgf = self.dgf_i[i]
            self.dgf_x = pd.Series(
                np.diag(self.A_i[i]), index=self.x_vars
            )
        else:
            self.convI = -9999
        return self.converged

    def testLinearity(self):
        """
        test whether the solution is moderately linear following chapter
        5.1 of Rodgers 2000.
        values lower than 1 indicate that the effect of linearization is
        smaller than the measurement error and problem is nearly linear.
        Populates self.nonlinearity.

        Parameters
        ----------

        Returns
        -------
        self.nonlinearity: float
          ratio of error due to linearization to measurement error. Should be
          below 1.
        self.trueNonlinearity: float
          As self.nonlinearity, but based on the true atmospheric state
          'self.x_truth'.
        """
        self.nonlinearity = np.zeros(self.x_n)*np.nan
        self.trueNonlinearity = np.nan
        if not self.converged:
            print("did not converge")
            return self.nonlinearity, self.trueNonlinearity
        lamb, II = np.linalg.eig(self.S_aposterior_i[self.convI])
        S_Ep_inv = _invertMatrix(np.array(self.S_y))
        lamb[np.isclose(lamb, 0)] = 0
        if np.any(lamb < 0):
            print(
                "found negative eigenvalues of S_aposterior_i, S_aposterior_i"
                " not semipositive definite!")
            return self.nonlinearity, self.trueNonlinearity
        error_pattern = lamb**0.5 * II
        for hh in range(self.x_n):
            x_hat = self.x_i[self.convI] + \
                error_pattern[:, hh]  # estimated truth
            xb_hat = pd.concat((x_hat, self.b_p))
            y_hat = self.forward(xb_hat, **self.forwardKwArgs)
            del_y = (y_hat - self.y_i[self.convI] - self.K_i[self.convI].dot(
                (x_hat - self.x_i[self.convI]).values))
            self.nonlinearity[hh] = del_y.T.dot(S_Ep_inv).dot(del_y)

        if self.x_truth is not None:
            xb_truth = pd.concat((self.x_truth, self.b_p))
            y_truth = self.forward(xb_truth, **self.forwardKwArgs)
            del_y = (y_truth - self.y_i[self.convI] - self.K_i[self.convI].dot(
                (self.x_truth - self.x_i[self.convI]).values))
            self.trueNonlinearity = del_y.T.dot(S_Ep_inv).dot(del_y)

        return self.nonlinearity, self.trueNonlinearity

    def chiSquareTest(self, significance=0.05):
        """
        test with significance level 'significance' whether retrieval agrees
        with measurements (see chapter 12.3.2 of Rodgers, 2000)

        Parameters
        ----------
        significance  : real, optional
          significance level, defaults to 0.05, i.e. probability is 5% that
           correct null hypothesis is rejected.

        Returns
        -------
        self.chi2Passed : bool
          True if chi² test passed, i.e. OE  retrieval agrees with
          measurements and null hypothesis is NOT rejected.
        self.chi2 : real
          chi² value
        self.chi2Test : real
          chi²  cutoff value with significance 'significance'

        """
        self.chi2Passed = False
        self.chi2 = np.nan
        self.chi2Test = np.nan

        if not self.converged:
            print("did not converge")
            return self.chi2Passed, self.chi2, self.chi2Test

        # Rodgers eq. 12.9
        S_deyd = self.S_y.values.dot(_invertMatrix(self.K_i[self.convI
                                                            ].values.dot(
            self.S_a.values.dot(self.K_i[self.convI].values.T)) + self.S_y
        )).dot(self.S_y.values)
        delta_y = self.y_i[self.convI] - self.y_obs
        self.chi2 = delta_y.T.dot(_invertMatrix(S_deyd)).dot(delta_y)
        self.chi2Test = scipy.stats.chi2.isf(significance, self.y_n)

        self.chi2Passed = self.chi2 <= self.chi2Test

        return self.chi2Passed, self.chi2, self.chi2Test

    def saveResults(self, fname):
        r'''
        Helper function to save a pyOptimalEstimation object. The forward
        operator is removed from the pyOptimalEstimation object before saving.

        Parameters
        ----------
        fname : str
          filename

        Returns
        -------
        None
        '''
        oeDict = deepcopy(self.__dict__)
        if "forward" in oeDict.keys():
            oeDict.pop("forward")
        np.save(fname, oeDict)
        return

    def plotIterations(
        self,
        cmap='viridis',
        figsize=(8, 10),
    ):
        r'''
        Plot the retrieval results using 4 panels: (1) iterations of x
        (normalized to self.x_truth or x[0]), (2) iterations of y (normalized to
        y_obs), (3) iterations of degrees of freedom, (4) iterations of
        convergence criteria

        Parameters
        ----------
        fileName : str, optional
          plot is saved to fileName, if provided
        cmap : str, optional
          colormap for 1st and 2nd panel (default 'hsv')
        figsize : tuple, optional
          Figure size in inch (default (8, 10))

        Returns
        -------
        matplotlib figure object
          The created figure.
        '''
        fig, [sp1, sp2, sp3, sp4] = plt.subplots(figsize=figsize, nrows=4,
                                                 sharex=True)
        d_i2 = np.array(self.d_i2)
        dgf_i = np.array(self.dgf_i)

        try:
            gamma = np.array(self.gam_i)
            noGam = len(gamma[gamma != 1])
            ind = np.argmin(d_i2[noGam:]) + noGam - 1
        except:
            ind = 0

        if self.converged:
            fig.suptitle('Sucessfully converged. Convergence criterion: %.3g'
                         ' Degrees of freedom: %.3g' % (d_i2[ind], dgf_i[ind]))
        else:
            fig.suptitle('Not converged. Convergence criterion: %.3g  Degrees'
                         ' of freedom: %.3g' % (d_i2[ind], dgf_i[ind]))

        colors = _niceColors(len(self.x_i[0].keys()), cmap=cmap)
        for kk, key in enumerate(self.x_i[0].keys()):
            xs = list()
            for xx in self.x_i[:-1]:
                xs.append(xx[key])
            if self.x_truth is not None:
                xs.append(self.x_truth[key])
                xs = np.array(xs) / self.x_truth[key]
            else:
                xs = np.array(xs) / xs[0]
            sp1.plot(xs, label=key, color=colors[kk])
        leg = sp1.legend(loc="best",
                         prop=font_manager.FontProperties(size=8))
        leg.get_frame().set_alpha(0.5)
        # sp1.set_xlabel("iteration")
        if self.x_truth is not None:
            sp1.set_ylabel("x-values\n(normalized to truth)")
        else:
            sp1.set_ylabel("x-values\n(normalized to prior)")

        sp1.axvline(ind, color="k")
        sp1.axvline(len(self.x_i)-2, ls=":", color="k")

        colors = _niceColors(len(self.y_i[0].keys()), cmap=cmap)
        for kk, key in enumerate(self.y_i[0].keys()):
            ys = list()
            for yy in self.y_i:
                ys.append(yy[key])
            ys.append(self.y_obs[key])
            ys = np.array(ys) / ys[-1]
            sp2.plot(ys, label=key, color=colors[kk])
        leg = sp2.legend(loc="best",
                         prop=font_manager.FontProperties(size=8))
        leg.get_frame().set_alpha(0.5)
        sp2.set_ylabel("y-values\n(normalized to measurement)")
        sp2.axvline(ind, color="k")
        sp2.axvline(len(self.x_i)-2, ls=":", color="k")

        sp3.plot(dgf_i, label="degrees of freedom")
        sp3.set_ylabel("degrees of freedom")
        sp3.axvline(len(self.x_i)-2, ls=":", color="k")
        sp3.axvline(ind, color="k")

        sp4.plot(d_i2, label="d_i2")
        sp4.set_xlabel("iteration")
        sp4.set_ylabel("convergence criterion")
        fig.subplots_adjust(hspace=0.1)
        sp4.set_xlim(0, len(self.x_i)-1)
        sp4.axvline(len(self.x_i)-2, ls=":", color="k")
        sp4.axvline(ind, color="k")
        sp4.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        xlabels = list(map(lambda x: "%i" % x, sp4.get_xticks()))
        xlabels[-1] = "truth/obs."
        sp4.set_xticklabels(xlabels)

        return fig

    def summary(self, returnXarray=False):
        '''Provide a summary of the retrieval results as a dictionary.

        Parameters
        ----------
        returnXarray : {bool}, optional
          return xarray dataset instead of dict. Can be easily combined when
          applying the retrieval multiple times. (the default is False)

        Returns
        -------
        dict or xarray.Dataset
          Summary of retrieval results
        '''

        if self.convI < 0:
            raise RuntimeError("Retrieval did not run successfully")

        summary = {}
        summary['x_a'] = self.x_a.rename_axis('x_vars')
        summary['S_a'] = self.S_a.rename_axis(
            'x_vars').rename_axis('x_vars', 1)
        summary['x_op'] = self.x_op.rename_axis('x_vars')
        summary['x_op_err'] = self.x_op_err.rename_axis('x_vars')
        summary['S_op'] = self.S_op.rename_axis(
            'x_vars').rename_axis('x_vars', 1)
        summary['dgf_x'] = self.dgf_x.rename_axis('x_vars')
        summary['y_obs'] = self.y_obs.rename_axis('y_vars')
        summary['S_y'] = self.S_y.rename_axis(
            'y_vars').rename_axis('y_vars', 1)

        summary['y_op'] = self.y_op.rename_axis('y_vars')
        if self.x_truth is not None:
            summary['x_truth'] = self.x_truth.rename_axis('x_vars')

        if len(self.b_vars) > 0:
            summary['b_p'] = self.b_p.rename_axis('b_vars')
            summary['S_b'] = self.S_b.rename_axis(
                'b_vars').rename_axis('b_vars', 1)

        if hasattr(self, 'nonlinearity'):
            summary['nonlinearity'] = self.nonlinearity
        if hasattr(self, 'trueNonlinearity'):
            summary['trueNonlinearity'] = self.trueNonlinearity
        if hasattr(self, 'chi2'):
            summary['chi2'] = self.chi2
            summary['chi2Test'] = self.chi2Test
            summary['chi2Passed'] = self.chi2Passed

        summary['dgf'] = self.dgf_i[self.convI]
        summary['convergedIteration'] = self.convI

        if returnXarray:
            import xarray as xr
            summary = xr.Dataset(summary)

        return summary


def optimalEstimation_loadResults(fname):
    r'''
    Helper function to load a saved pyOptimalEstimation object

    Parameters
    ----------
    fname : str
      filename

    Returns
    -------
    pyOptimalEstimation object
      pyOptimalEstimation obtained from file.
    '''
    oeDict = np.load(fname)
    oe = _oeDict2Object(oeDict.tolist())
    return oe


def _oeDict2Object(oeDict):
    r'''
    Helper function to convert a oe-dictionary (usually loaded from a file) to
    a pyOptimalEstimation object

    Parameters
    ----------
    oeDict : dict
      dictionary object

    Returns
    -------
    pyOptimalEstimation object
      pyOptimalEstimation object obtained from file.
    '''
    oe = optimalEstimation(
        oeDict.pop("x_vars"),
        oeDict.pop("x_a"),
        oeDict.pop("S_a"),
        oeDict.pop("y_vars"),
        oeDict.pop("S_y"),
        oeDict.pop("y_obs"),
        None
    )
    for kk in oeDict.keys():
        oe.__dict__[kk] = oeDict[kk]
    return oe


def _niceColors(length, cmap='hsv'):
    r'''
    Helper function to provide colors for plotting

    Parameters
    ----------
    length : int
      The number of required colors
    cmap : str, optional
      Matplotlib colormap. Defaults to hsv.

    Returns
    -------
    list of colorcodes
      list of colors
    '''
    colors = list()
    cm = plt.get_cmap(cmap)
    for l in range(length):
        colors.append(cm(1.*l/length))
    return colors


def _invertMatrix(A):
    '''
    Wrapper funtion for np.linalg.inv, because original function reports
    LinAlgError if nan in array for some numpy versions. We want that the
    retrieval is robust with respect to that
    '''
    if np.any(np.isnan(A)):
        warnings.warn("Found nan in Matrix during inversion", UserWarning)
        return np.zeros_like(A) * np.nan
    else:
        return np.linalg.inv(A)
