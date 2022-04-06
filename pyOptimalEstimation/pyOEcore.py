# -*- coding: utf-8 -*-

# pyOptimalEstimation

# Copyright (C) 2014-21 Maximilian Maahn, Leipzig University
# maximilian.maahn@uni-leipzig.de
# https://github.com/maahn/pyOptimalEstimation

# With contributions by M. Echeverri

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import time
from copy import deepcopy
import warnings

import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.ticker as ticker
import pandas as pd


class optimalEstimation(object):
    r'''
    The core optimalEstimation class, which contains all required parameters.
    See [1]_ for an extensive introduction into Optimal Estimation theory, 
    [2]_ discusses this library

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
    userJacobian : function, optional
        For forwarld models that can calculate the Jacobian internally (e.g.
        RTTOV), a call to estiamte the Jacobian can be added. Otherwise, the 
        Jacobian is estimated by pyOEusing the standard 'forward' call. The 
        fucntion is expected as ``self.userJacobian(xb, self.perturbation, \
        self.y_vars, **self.forwardKwArgs): return jacobian``
        with xb = pd.concat((x,b)). Defaults to None
    x_truth : pd.Series or list or np.ndarray, optional
        If truth of state x is known, it can added to the data object. If
        provided, the value will be used for the routines linearityTest and
        plotIterations, but _not_ by the retrieval itself. Defaults to None/
    b_vars : list of str, optional
        names of the elements of parameter vector b. Defaults to [].
    b_p : pd.Series or list or np.ndarray.
        parameter vector b.  defaults to []. Note that defining b_p makes
        only sence if S_b != 0. Otherwise it is easier (and cheaper) to
        hardcode b into the forward operator.
    S_b : pd.DataFrame or list or np.ndarray
        covariance matrix of parameter b. Defaults to [[]].
    forwardKwArgs : dict,optional
        additional keyword arguments for ``forward`` function.
    multipleForwardKwArgs : dict,optional
        additional keyword arguments for forward function in case multiple 
        profiles should be provided to the forward operator at once. If not .
        defined, ``forwardKwArgs`` is used instead and ``forward`` is called 
        for every profile separately
    x_lowerLimit : dict, optional
        reset state vector x[key] to x_lowerLimit[key] in case x_lowerLimit is
        undercut. Defaults to {}.
    x_upperLimit : dict, optional
        reset state vector x[key] to x_upperLimit[key] in case x_upperLimit is
        exceeded. Defaults to {}.
    perturbation : float or dict of floats, optional
        relative perturbation of statet vector x to estimate the Jacobian. Can
        be specified for every element of x seperately. Defaults to 0.1 of
        prior.
    disturbance : float or dict of floats, optional
        DEPRECATED: Identical to ``perturbation`` option. If both option are 
        provided, ``perturbation``  is used instead. 
    useFactorInJac : bool,optional
        True if disturbance should be applied by multiplication, False if it
        should by applied by addition of fraction of prior. Defaults to False.
    gammaFactor : list of floats, optional
        Use additional gamma parameter for retrieval, see [3]_.
    convergenceTest : {'x', 'y', 'auto'}, optional
        Apply convergence test in x or y-space. If 'auto' is 
        selected, the test will be done in x-space if len(x) <= len(y) and in 
        y-space otherwise. Experience shows that in both cases convergence is 
        faster in x-space without impacting retrieval quality. Defaults to 'x'.
    convergenceFactor : int, optional
        Factor by which the convergence criterion needs to be smaller than
        len(x) or len(y) 
    verbose: bool, optional
        True or not present: iteration, residual, etc. printed to screen during
        normal operation. If False, it will turn off such notifications.     

    Attributes
    ----------
    converged : boolean
      True if retriveal converged successfully
    x_op : pd.Series
      optimal state given the observations, i.e. retrieval solution
    y_op : pd.Series
      Optimal y, i.e. observation associated with retrieval solution
    S_op : pd.DataFrame
      covariance of x_op, i.e. solution uncertainty
    x_op_err : pd.Series
      1 sigma errors of x_op. derived with sqrt(diag(S_op))
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
    S_aposteriori_i : list of pd.DataFrame
      a posteriori covariance matrix of x for each iteration
    gam_i : list of floats
      gamma parameters used in retrievals, see also `gammaFactor` and  [1]_.
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
    .. [1] Rodgers, C. D., 2000: Inverse Methods for Atmospheric Sounding:
    Theory and Practice. World Scientific Publishing Company, 240 pp. 
    https://library.wmo.int/index.php?lvl=notice_display&id=12279.

    .. [2] Maahn, M., D. D. Turner, U. Löhnert, D. J. Posselt, K. Ebell, G. 
    G. Mace, and J. M. Comstock, 2020: Optimal Estimation Retrievals and Their
    Uncertainties: What Every Atmospheric Scientist Should Know. Bull. Amer. 
    Meteor. Soc., 101, E1512–E1523, https://doi.org/10.1175/BAMS-D-19-0027.1.

    .. [3] Turner, D. D., and U. Löhnert, 2014: Information Content and
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
                 userJacobian=None,
                 x_truth=None,
                 b_vars=[],
                 b_p=[],
                 S_b=[[]],
                 x_lowerLimit={},
                 x_upperLimit={},
                 useFactorInJac=False,
                 gammaFactor=None,
                 perturbation=0.1,
                 disturbance=None,
                 convergenceFactor=10,
                 convergenceTest='x',
                 forwardKwArgs={},
                 multipleForwardKwArgs=None,
                 verbose=None
                 ):

        # some initital tests
        assert np.linalg.matrix_rank(S_a) == S_a.shape[-1],\
            'S_a must not be singular'
        assert np.linalg.matrix_rank(S_y) == S_y.shape[-1],\
            'S_y must not be singular'
        for inVar in [x_a, S_a, S_y, y_obs]:
            assert not np.any(np.isnan(inVar))

        self.x_vars = list(x_vars)
        self.x_a = pd.Series(x_a, index=self.x_vars)
        self.S_a = pd.DataFrame(
            S_a, index=self.x_vars, columns=self.x_vars)
        self.x_a_err = np.sqrt(
            pd.Series(np.diag(self.S_a), index=self.x_vars)
        )
        self.x_n = len(self.x_vars)
        self.y_vars = list(y_vars)
        self.S_y = pd.DataFrame(
            S_y, index=self.y_vars, columns=self.y_vars)
        self.y_obs = pd.Series(y_obs, index=self.y_vars)
        self.y_n = len(self.y_vars)
        self.forward = forward
        self.userJacobian = userJacobian
        self.x_truth = pd.Series(x_truth, index=self.x_vars, dtype=np.float64)

        # We want to save at least the name because the forward function
        # is removed for saving
        try:
            self.forward_name = forward.__name__
        except AttributeError:
            self.forward_name = None
        try:
            self.userJacobian_name = userJacobian.__name__  
        except AttributeError:
            self.userJacobian_name = None

        self.b_vars = list(b_vars)
        self.b_n = len(self.b_vars)
        assert self.b_n == len(b_p)
        self.b_p = pd.Series(b_p, index=self.b_vars, dtype=np.float64)
        self.S_b = pd.DataFrame(
            S_b, index=self.b_vars, columns=self.b_vars)
        self.b_p_err = np.sqrt(
            pd.Series(np.diag(self.S_b), index=self.b_vars)
        )
        self.forwardKwArgs = forwardKwArgs
        self.multipleForwardKwArgs = multipleForwardKwArgs
        self.verbose = verbose
        self.x_lowerLimit = x_lowerLimit
        self.x_upperLimit = x_upperLimit
        self.useFactorInJac = useFactorInJac
        self.gammaFactor = gammaFactor
        if disturbance is not None:
            self.perturbation = disturbance
            print('Warning. The option "disturbance" is deprecated, use '
                  '"perturbation" instead')
        self.perturbation = perturbation
        self.convergenceFactor = convergenceFactor
        self.convergenceTest = convergenceTest

        self.converged = False
        self.K_i = None
        self.x_i = None
        self.y_i = None
        self.dgf_i = None
        self.A_i = None
        self.d_i2 = None
        self.S_aposteriori_i = None
        self.gam_i = None
        self.convI = None
        self.x_op = None
        self.y_op = None
        self.S_op = None
        self.x_op_err = None
        self.dgf = None
        self.dgf_x = None

        self._y_a = None

        return

    def getJacobian(self, xb, y):
        r'''
        Author: M. Echeverri, May 2021.

        estimate Jacobian using the forward model and the specified 
        perturbation

        Parameters
        ----------
        xb  : pd.Series or list or np.ndarray
          combination of state vector x and parameter vector b
        y : pd.Series or list or np.ndarray
          measurement vector for xb

        Returns
        -------
        pd.DataFrame
          Jacobian around x
        pd.DataFrame
          Jacobian around b
        '''

        xb_vars = self.x_vars + self.b_vars
        # xb = pd.Series(xb, index=xb_vars, dtype=float)
        xb_err = pd.concat((self.x_a_err, self.b_p_err))
        # y = pd.Series(y, index=self.y_vars, dtype=float)

        # If a factor is used to perturb xb, xb must not be zero.
        assert not (self.useFactorInJac and np.any(xb == 0))

        if type(self.perturbation) == float:
            perturbations = dict()
            for key in xb_vars:
                perturbations[key] = self.perturbation
        elif type(self.perturbation) == dict:
            perturbations = self.perturbation
        else:
            raise TypeError("perturbation must be type dict or float")

        # perturbations == perturbation, but perturbation is only a numpy array
        # order in elements of "perturbation" follows xb_vars.
        # Question to MM: does perturbations need to be dict?
        perturbation = np.zeros((len(xb_vars),), dtype=np.float64)
        i = 0
        for key, value in perturbations.items():
            perturbation[i] = value
            i += 1

        perturbedKeys = []
        for tup in xb_vars:
            perturbedKeys.append("perturbed %s" % tup)

        # Numpy array, dims: ("perturbedKeys","xb_bars"); "perturbedKeys" = "perturbed "+"xb_vars"
        # Initialize to xb in rows:
        aux_xb_perturbed = np.ones((len(xb_vars), len(xb_vars)), dtype=np.float64) *\
            xb.to_numpy().reshape(1, len(xb_vars))

        if self.useFactorInJac:
            np.fill_diagonal(aux_xb_perturbed,
                             (np.diag(aux_xb_perturbed) * perturbation))
        else:
            np.fill_diagonal(aux_xb_perturbed,
                             (np.diag(aux_xb_perturbed) + (xb_err.to_numpy()*perturbation)))

        self.xb_perturbed = pd.DataFrame(aux_xb_perturbed,
                                         columns=xb_vars, index=perturbedKeys, dtype=np.float64)

        # Calculate dy : the forward model for all the perturbed profiles

        if self.multipleForwardKwArgs != None:  # if forward arguments for multiple profiles are provided
            # This version exploits the advantage of calling the forward model for several
            # atmospherical profiles at the same time. In doing so the code fully uses any
            # "under the hood" optimizations inside the "forward" function.

            # Extra function arguments are needed (**self.multipleForwardKwArgs) to pass by the multiple-profiles configuration:
            # for using a forward solver like CRTM or RTTOV, a specific instrument and profile configuration
            # is needed to include the fact that more than 1 profile will be simulated in the same call. Notice
            # that this DOES NOT mean that the "forward" model needs to be modified (this is not required)

            aux_y_perturbed = self.forward(
                self.xb_perturbed.T, **self.multipleForwardKwArgs)
            # Assemble y_perturbed Dataframe to keep consistency the format of the object.
            self.y_perturbed = pd.DataFrame(aux_y_perturbed.T,
                                            columns=self.y_vars,
                                            index=perturbedKeys,
                                            dtype=np.float64
                                            )

        # if forward arguments for multiple profiles are NOT provided   (previous version)
        else:
            self.y_perturbed = pd.DataFrame(
                columns=self.y_vars,
                index=perturbedKeys,
                dtype=np.float64
            )
            for xb_dist in self.xb_perturbed.index:
                self.y_perturbed.loc[xb_dist] = self.forward(
                    self.xb_perturbed.loc[xb_dist], **self.forwardKwArgs)

            # This line only if not using the multiple calls
            aux_y_perturbed = self.y_perturbed.to_numpy().T
            # from Forward model but still using the broadcast Jacobian calculation

        # Calc Jacobian New:

        # row vector containing observations (y)
        obs = y.to_numpy(dtype=np.float64).reshape(1, len(self.y_vars))

        # Compute dx (i.e. distance to perturbed parameters)

        if self.useFactorInJac:
            aux_dist = (xb.to_numpy(dtype=np.float64)*(perturbation-1.0))
        else:
            aux_dist = (perturbation * xb_err.to_numpy(dtype=np.float64))

        # Check there are no zero distances:

        assert np.sum((aux_dist == 0)) == 0, 'S_a&s_b must not contain zeros on '\
            'diagonal'

        # If assertion pass, then compute the inverse of distance and reshape it into a column vector:

        # column vector
        inv_dist = (1/aux_dist).reshape(len(xb_err.to_numpy()), 1)

        # Use Numpy broadcasting rules to efficiently compute the Jacobian

        aux_jacobian = (aux_y_perturbed.T - obs) * inv_dist  # Numpy broadcast

        # Assemble Jacobian Dataframe:

        jacobian = pd.DataFrame(aux_jacobian.T,
                                index=self.y_vars, columns=perturbedKeys)

        jacobian[np.isnan(jacobian) | np.isinf(jacobian)] = 0.
        jacobian_x = jacobian[["perturbed %s" % s for s in self.x_vars]]
        jacobian_b = jacobian[["perturbed %s" % s for s in self.b_vars]]

        return jacobian_x, jacobian_b

    def getJacobian_external(self, xb, y):
        r'''
        Author: M. Echeverri, June 2021.

        estimate Jacobian using the external function provided by user and 
        the specified perturbation. This method has external dependencies 

        Parameters
        ----------
        xb  : pd.Series or list or np.ndarray
          combination of state vector x and parameter vector b
        y : pd.Series or list or np.ndarray
          measurement vector for xb

        Returns
        -------
        pd.DataFrame
          Jacobian around x
        pd.DataFrame
          Jacobian around b
        '''

        xb_vars = self.x_vars + self.b_vars
        # xb = pd.Series(xb, index=xb_vars, dtype=float)
        xb_err = pd.concat((self.x_a_err, self.b_p_err))
        # y = pd.Series(y, index=self.y_vars, dtype=float)

        # If a factor is used to perturb xb, xb must not be zero.
        assert not (self.useFactorInJac and np.any(xb == 0))

        if type(self.perturbation) == float:
            perturbations = dict()
            for key in xb_vars:
                perturbations[key] = self.perturbation
        elif type(self.perturbation) == dict:
            perturbations = self.perturbation
        else:
            raise TypeError("perturbation must be type dict or float")

        # perturbations == perturbation, but perturbation is only a numpy array
        # order in elements of "perturbation" follows xb_vars.

        perturbation = np.zeros((len(xb_vars),), dtype=np.float64)
        i = 0
        for key, value in perturbations.items():
            perturbation[i] = value
            i += 1

        perturbedKeys = []
        for tup in xb_vars:
            perturbedKeys.append("perturbed %s" % tup)

        # Compute dx (i.e. distance to perturbed parameters)

        if self.useFactorInJac:
            aux_dist = (xb.to_numpy()*(perturbation-1.0))
        else:
            aux_dist = (perturbation * xb_err.to_numpy())

        # Compute Jacobian using user's Jacobian function

        jac_numpy = self.userJacobian(xb, self.perturbation,
                                      self.y_vars, **self.forwardKwArgs)

        # Assemble  Jacobian Dataframe:

        jacobian = pd.DataFrame(jac_numpy,
                                index=self.y_vars, columns=perturbedKeys)

        jacobian[np.isnan(jacobian) | np.isinf(jacobian)] = 0.
        jacobian_x = jacobian[["perturbed %s" % s for s in self.x_vars]]
        jacobian_b = jacobian[["perturbed %s" % s for s in self.b_vars]]

        # to deprecate? (assertions are present in other parts of pyOpEst,
        #    so this is added here for compliance with those assertions):

        self.xb_perturbed = pd.DataFrame(
            columns=xb_vars, index=perturbedKeys, dtype=float)
        self.y_perturbed = pd.DataFrame(
            columns=self.y_vars,
            index=perturbedKeys,
            dtype=np.float64
        )

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

        self.K_i = [0]*maxIter  # list of jacobians
        self.K_b_i = [0]*maxIter  # list of jacobians for parameter vector
        self.x_i = [0]*(maxIter+1)
        self.y_i = [0]*(maxIter+1)
        self.dgf_i = [0]*maxIter
        self.H_i = [0]*maxIter  # Shannon information content
        self.A_i = [0]*maxIter
        self.d_i2 = [0]*maxIter  # convergence criteria
        self.S_ep_i = [0] * maxIter
        self.S_aposteriori_i = [0] * maxIter
        # self.Pxy_i = [0] *maxIter
        self.gam_i = [1]*maxIter

        S_a = np.array(self.S_a)  # Covariance of prior estimate of x
        assert np.all(S_a == S_a.T), 'S_a must be symmetric'
        S_a_inv = invertMatrix(S_a)  # S_a inverted

        if self.gammaFactor:
            assert len(self.gammaFactor) <= maxIter
            self.gam_i[:len(self.gammaFactor)] = self.gammaFactor

        # treat first guess
        if x_0 is None:
            self.x_i[0] = self.x_a
        else:
            self.x_i[0] = pd.Series(x_0, index=self.x_vars)

        # y of first guess
        xb_i0 = pd.concat((self.x_i[0], self.b_p))
        y = self.forward(xb_i0, **self.forwardKwArgs)
        self.y_i[0] = pd.Series(y, index=self.y_vars, dtype=float)

        for i in range(maxIter):

            if (self.userJacobian != None):  # then user's Jacobian function is used

                self.K_i[i], self.K_b_i[i] = self.getJacobian_external(
                    pd.concat((self.x_i[i], self.b_p)), self.y_i[i])

            else:                           # uses method getJacobian

                self.K_i[i], self.K_b_i[i] = self.getJacobian(
                    pd.concat((self.x_i[i], self.b_p)), self.y_i[i])

            if np.sum(self.S_b.shape) > 0:
                S_ep_b = self.K_b_i[i].values.dot(
                    self.S_b.values).dot(self.K_b_i[i].values.T)
            else:
                S_ep_b = 0
            # S_epsilon Covariance of measurement noise including parameter
            # uncertainty (Rodgers, sec 3.4.3)
            self.S_ep_i[i] = self.S_y.values + S_ep_b

            # make sure S_y and S_ep are symmetric
            assert np.all(self.S_y.values == self.S_y.values.T), \
                'S_y must be symmetric'
            assert np.isclose(self.S_ep_i[i], self.S_ep_i[i].T).all(), \
                'S_ep must be symmetric'

            # S_ep inverted
            S_ep_inv = invertMatrix(self.S_ep_i[i])

            assert np.all(self.y_perturbed.keys() == self.S_y.keys())
            assert np.all(self.S_y.keys() == self.K_i[i].index)
            assert np.all(self.S_a.index == self.x_a.index)
            assert np.all(self.x_a.index.tolist(
            )+self.b_p.index.tolist() == self.xb_perturbed.columns)
            assert np.all(self.xb_perturbed.index.tolist(
            ) == self.K_i[i].columns.tolist()+self.K_b_i[i].columns.tolist())

            K = np.array(self.K_i[i])

            # reformulated using Turner and Löhnert 2013:
            B = (self.gam_i[i] * S_a_inv) + \
                K.T.dot(S_ep_inv.dot(K))  # eq 3
            B_inv = invertMatrix(B)
            self.S_aposteriori_i[i] = B_inv.dot(
                (self.gam_i[i]**2 * S_a_inv) + K.T.dot(S_ep_inv.dot(K))
            ).dot(B_inv)  # eq2

            self.S_aposteriori_i[i] = pd.DataFrame(
                self.S_aposteriori_i[i],
                index=self.x_a.index,
                columns=self.x_a.index
            )
            G = B_inv.dot(K.T.dot(S_ep_inv))
            self.A_i[i] = G.dot(K)  # eq 4

            # estimate next x
            self.x_i[i+1] = self.x_a +\
                B_inv.dot(
                K.T.dot(S_ep_inv.dot(self.y_obs - self.y_i[i] +
                                     K.dot(self.x_i[i] - self.x_a))))  # eq 1

            # estimate next y
            xb_i1 = pd.concat((self.x_i[i+1], self.b_p))
            y = self.forward(xb_i1, **self.forwardKwArgs)
            self.y_i[i+1] = pd.Series(y, index=self.y_vars, dtype=float)

            self.dgf_i[i] = np.trace(self.A_i[i])
            # eq. 2.80 Rodgers
            self.H_i[i] = -0.5 * \
                np.log(np.linalg.det(np.identity(self.x_n) - self.A_i[i]))

            # check whether i+1 is valid
            for jj, xKey in enumerate(self.x_vars):
                if (xKey in self.x_lowerLimit.keys()) and (
                        self.x_i[i+1].iloc[jj] < self.x_lowerLimit[xKey]):
                    print("#"*60)
                    print("reset due to x_lowerLimit: %s from %f to %f in "
                          "iteration %d" % (
                              xKey,
                              self.x_i[i+1].iloc[jj],
                              self.x_a.iloc[jj], i
                          ))
                    self.x_i[i+1].iloc[jj] = self.x_a.iloc[jj]
                if (xKey in self.x_upperLimit.keys()) and (
                        self.x_i[i+1].iloc[jj] > self.x_upperLimit[xKey]):
                    print("#"*60)
                    print("reset due to x_upperLimit: %s from %f to %f in "
                          "iteration %d" % (
                              xKey,
                              self.x_i[i+1].iloc[jj],
                              self.x_a.iloc[jj], i
                          ))
                    self.x_i[i+1].iloc[jj] = self.x_a.iloc[jj]
                if np.isnan(self.x_i[i+1].iloc[jj]):
                    print("#"*60)
                    print("reset due to nan: %s from %f to %f in iteration "
                          "%d" % (
                              xKey,
                              self.x_i[i+1].iloc[jj],
                              self.x_a.iloc[jj], i
                          ))
                    self.x_i[i+1].iloc[jj] = self.x_a.iloc[jj]

            # test in x space for len(y) > len(x)
            if (
                (self.convergenceTest == 'x')
                or
                ((self.convergenceTest == 'auto') and (self.x_n <= self.y_n))
            ):
                # convergence criterion eq 5.29 Rodgers 2000
                dx = self.x_i[i] - self.x_i[i+1]
                self.d_i2[i] = dx.T.dot(invertMatrix(
                    self.S_aposteriori_i[i])).dot(dx)
                d_i2_limit = self.x_n/float(self.convergenceFactor)
                usingTest = 'x-space'
            # test in y space for for len(y) < len(x)
            elif (
                (self.convergenceTest == 'y')
                or
                (self.convergenceTest == 'auto')
            ):
                # convergence criterion eqs 5.27 &  5.33 Rodgers 2000
                dy = self.y_i[i+1] - self.y_i[i]
                KSaKSep = K.dot(S_a).dot(K.T) + self.S_ep_i[i]
                KSaKSep_inv = invertMatrix(KSaKSep)

                S_deyd = self.S_ep_i[i].dot(KSaKSep_inv).dot(self.S_ep_i[i])

                self.d_i2[i] = dy.T.dot(invertMatrix(
                    S_deyd)).dot(dy)
                d_i2_limit = self.y_n/float(self.convergenceFactor)
                usingTest = 'y-space'
            else:
                raise ValueError('Do not understand convergenceTest %s' %
                                 self.convergenceTest)

            assert not self.d_i2[i] < 0, 'a negative convergence cirterion'
            ' means someting has gotten really wrong'

            # stop if we converged in the step before
            if self.converged:
                if(self.verbose != None):
                        if(self.verbose):
                            print("%.2f s, iteration %i, degrees of freedom: %.2f of %i, "
                                  "done.  %.3f" % (
                                      time.time()-startTime, i, self.dgf_i[i], self.x_n,
                                      self.d_i2[i]))          
                else:
                    print("%.2f s, iteration %i, degrees of freedom: %.2f of %i, "
                                  "done.  %.3f" % (
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
                if (np.abs(self.d_i2[i]) < d_i2_limit) and (
                        self.gam_i[i] == 1) and (self.d_i2[i] != 0):
                    if(self.verbose != None):
                        if(self.verbose):    
                            print("%.2f s, iteration %i, degrees of freedom: %.2f of"
                                  " %i, converged (%s):  %.3f" % (
                                     time.time() -
                                      startTime, i, self.dgf_i[i], self.x_n,
                                      usingTest, self.d_i2[i]))
                    else:
                        print("%.2f s, iteration %i, degrees of freedom: %.2f of"
                                  " %i, converged (%s):  %.3f" % (
                                     time.time() -
                                      startTime, i, self.dgf_i[i], self.x_n,
                                      usingTest, self.d_i2[i]))                  
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
                    if(self.verbose != None):
                        if(self.verbose):
                            print("%.2f s, iteration %i, degrees of freedom:"
                                  " %.2f of %i, not converged (%s): "
                                  " %.3f" % (
                                      time.time()-startTime, i, self.dgf_i[i],
                                      self.x_n, usingTest, self.d_i2[i]))
                    else:
                        print("%.2f s, iteration %i, degrees of freedom:"
                                  " %.2f of %i, not converged (%s): "
                                  " %.3f" % (
                                      time.time()-startTime, i, self.dgf_i[i],
                                      self.x_n, usingTest, self.d_i2[i]))                  

            #print("%.2f s , TimeRest" % (time.time()-startTimeRest))

        self.K_i = self.K_i[:i+1]
        self.K_b_i = self.K_b_i[:i+1]
        self.x_i = self.x_i[:i+2]
        self.y_i = self.y_i[:i+2]
        self.dgf_i = self.dgf_i[:i+1]
        self.A_i = self.A_i[:i+1]
        self.H_i = self.H_i[:i+1]
        self.d_i2 = self.d_i2[:i+1]
        self.S_ep_i = self.S_ep_i[:i+1]

        self.S_aposteriori_i = self.S_aposteriori_i[:i+1]

        self.gam_i = self.gam_i[:i+1]
        if self.converged:
            self.convI = i

            self.x_op = self.x_i[i]
            self.y_op = self.y_i[i]
            self.S_op = self.S_aposteriori_i[i]
            self.x_op_err = np.sqrt(
                pd.Series(np.diag(
                    self.S_aposteriori_i[self.convI]), index=self.x_vars)
            )
            self.dgf = self.dgf_i[i]
            self.dgf_x = pd.Series(
                np.diag(self.A_i[i]), index=self.x_vars
            )

        else:
            self.convI = -9999
            self.x_op = np.nan
            self.y_op = np.nan
            self.S_op = np.nan
            self.x_op_err = np.nan
            self.dgf = np.nan
            self.dgf_x = np.nan

        return self.converged

    @property
    def y_a(self):
        '''
        Estimate the observations corresponding to the prior.
        '''
        if self._y_a is None:
            xb_a = pd.concat((self.x_a, self.b_p))
            self._y_a = pd.Series(self.forward(xb_a, **self.forwardKwArgs),
                                  index=self.y_vars)
        return self._y_a

    def linearityTest(
        self,
        maxErrorPatterns=10,
        significance=0.05,
        atol=1e-5
    ):
        """
        test whether the solution is moderately linear following chapter
        5.1 of Rodgers 2000.
        values lower than 1 indicate that the effect of linearization is
        smaller than the measurement error and problem is nearly linear.
        Populates self.linearity.

        Parameters
        ----------
        maxErrorPatterns  : int, optional
          maximum number of error patterns to return. Provide None to return
        all.
        significance  : real, optional
          significance level, defaults to 0.05, i.e. probability is 5% that
           correct null hypothesis is rejected. Only used when testing 
           against x_truth.
        atol : float (default 1e-5)
            The absolute tolerance for comparing eigen values to zero. We 
            found that values should be than the numpy.isclose defualt value 
            of 1e-8.

        Returns
        -------
        self.linearity: float
          ratio of error due to linearization to measurement error sorted by 
          size. Should be below 1 for all.
        self.trueLinearityChi2: float
           Chi2 value that model is moderately linear based on 'self.x_truth'.
           Must be smaller than critical value to conclude thast model is
           linear.
        self.trueLinearityChi2Critical: float
           Corresponding critical Chi2 value. 
        """
        self.linearity = np.zeros(self.x_n)*np.nan
        self.trueLinearityChi2 = np.nan
        self.trueLinearityChi2Critical = np.nan

        if not self.converged:
            print("did not converge")
            return self.linearity, self.trueLinearity
        lamb, II = np.linalg.eig(self.S_aposteriori_i[self.convI])
        S_ep_inv = invertMatrix(np.array(self.S_ep_i[self.convI]))
        lamb[np.isclose(lamb, 0)] = 0
        if np.any(lamb < 0):
            print(
                "found negative eigenvalues of S_aposteriori_i, "
                " S_aposteriori_i not semipositive definite!")
            return self.linearity, self.trueLinearity
        error_pattern = lamb**0.5 * II
        for hh in range(self.x_n):
            x_hat = self.x_i[self.convI] + \
                error_pattern[:, hh]  # estimated truth
            xb_hat = pd.concat((x_hat, self.b_p))
            y_hat = self.forward(xb_hat, **self.forwardKwArgs)
            del_y = (y_hat - self.y_i[self.convI] - self.K_i[self.convI].dot(
                (x_hat - self.x_i[self.convI]).values))
            self.linearity[hh] = del_y.T.dot(S_ep_inv).dot(del_y)

        self.linearity = sorted(
            self.linearity, reverse=True)[slice(None, maxErrorPatterns)]

        if self.x_truth is not None:
            xb_truth = pd.concat((self.x_truth, self.b_p))
            y_truth = self.forward(xb_truth, **self.forwardKwArgs)
            del_y = (y_truth - self.y_i[self.convI] - self.K_i[self.convI].dot(
                (self.x_truth - self.x_i[self.convI]).values))
            self.trueLinearity = del_y.T.dot(S_ep_inv).dot(del_y)

            res = _testChi2(self.S_y.values, del_y, significance, atol)
            self.trueLinearityChi2, self.trueLinearityChi2Critical = res

        return self.linearity, self.trueLinearityChi2, \
            self.trueLinearityChi2Critical

    def chiSquareTest(self, significance=0.05):
        '''

        test with significance level 'significance' whether 
        A) optimal solution agrees with observation in Y space
        B) observation agrees with prior in Y space
        C) optimal solution agrees with prior in Y space
        D) optimal solution agrees with priot in X space

        Parameters
        ----------
        significance  : real, optional
          significance level, defaults to 0.05, i.e. probability is 5% that
           correct null hypothesis is rejected.

        Returns
        -------
        Pandas Series (dtype bool):
            True if test is passed
        Pandas Series (dtype float):
            Chi2 value for tests. Must be smaler than critical value to pass
            tests.
        Pandas Series (dtype float):
            Critical Chi2 value for tests
        '''
        chi2names = pd.Index([
            'Y_Optimal_vs_Observation',
            'Y_Observation_vs_Prior',
            'Y_Optimal_vs_Prior',
            'X_Optimal_vs_Prior',
        ], name='chi2test')

        chi2Cols = [
            'chi2value',
            'chi2critical',
        ]

        if not self.converged:
            print("did not converge")
            pd.DataFrame(
                np.zeros((4, 2)),
                index=chi2names,
                columns=chi2Cols,
            )*np.nan
        else:
            YOptimalObservation = self.chiSquareTestYOptimalObservation(
                significance=significance)
            YObservationPrior = self.chiSquareTestYObservationPrior(
                significance=significance)
            YOptimalPrior = self.chiSquareTestYOptimalPrior(
                significance=significance)
            XOptimalPrior = self.chiSquareTestXOptimalPrior(
                significance=significance)

            self.chi2Results = pd.DataFrame(
                np.array([
                    YOptimalObservation,
                    YObservationPrior,
                    YOptimalPrior,
                    XOptimalPrior,
                ]),
                index=chi2names,
                columns=chi2Cols,
            )

        passed = self.chi2Results['chi2value'] < self.\
            chi2Results['chi2critical']

        return passed, self.chi2Results['chi2value'], \
            self.chi2Results['chi2critical']

    def chiSquareTestYOptimalObservation(self, significance=0.05, atol=1e-5):
        """
        test with significance level 'significance' whether retrieval agrees
        with measurements (see chapter 12.3.2 of Rodgers, 2000)

        Parameters
        ----------
        significance  : real, optional
          significance level, defaults to 0.05, i.e. probability is 5% that
           correct null hypothesis is rejected.
        atol : float (default 1e-5)
            The absolute tolerance for comparing eigen values to zero. We 
            found that values should be than the numpy.isclose defualt value 
            of 1e-8.
        Returns
        -------
        chi2Passed : bool
          True if chi² test passed, i.e. OE  retrieval agrees with
          measurements and null hypothesis is NOT rejected.
        chi2 : real
          chi² value
        chi2TestY : real
          chi²  cutoff value with significance 'significance'

        """
        assert self.converged

        Sa = self.S_a.values
        Sep = self.S_ep_i[self.convI]
        K = self.K_i[self.convI].values

        # Rodgers eq. 12.9
        KSaKSep_inv = invertMatrix(K.dot(Sa).dot(K.T) + Sep)
        S_deyd = Sep.dot(KSaKSep_inv).dot(Sep)
        delta_y = self.y_i[self.convI] - self.y_obs

        chi2, chi2TestY = _testChi2(S_deyd, delta_y, significance, atol)

        return chi2, chi2TestY

    def chiSquareTestYObservationPrior(self, significance=0.05, atol=1e-5):
        """
        test with significance level 'significance' whether measurement agrees
        with prior (see chapter 12.3.3.1 of Rodgers, 2000)

        Parameters
        ----------
        significance  : real, optional
          significance level, defaults to 0.05, i.e. probability is 5% that
           correct null hypothesis is rejected.
        atol : float (default 1e-5)
            The absolute tolerance for comparing eigen values to zero. We 
            found that values should be than the numpy.isclose defualt value 
            of 1e-8.
        Returns
        -------
        YObservationPrior : bool
          True if chi² test passed, i.e. OE  retrieval agrees with
          measurements and null hypothesis is NOT rejected.
        YObservationPrior: real
          chi² value
        chi2TestY : real
          chi²  cutoff value with significance 'significance'

        """

        assert self.converged

        delta_y = self.y_obs - self.y_a
        Sa = self.S_a.values
        Sep = self.S_ep_i[self.convI]
        K = self.K_i[self.convI].values
        KSaKSep = K.dot(Sa).dot(K.T) + Sep

        chi2, chi2TestY = _testChi2(KSaKSep, delta_y, significance, atol)

        return chi2, chi2TestY

    def chiSquareTestYOptimalPrior(self, significance=0.05, atol=1e-5):
        """
        test with significance level 'significance' whether retrieval result agrees
        with prior in y space (see chapter 12.3.3.3 of Rodgers, 2000)

        Parameters
        ----------
        significance  : real, optional
          significance level, defaults to 0.05, i.e. probability is 5% that
           correct null hypothesis is rejected.
        atol : float (default 1e-5)
            The absolute tolerance for comparing eigen values to zero. We 
            found that values should be than the numpy.isclose defualt value 
            of 1e-8.

        Returns
        -------
        chi2Passe : bool
          True if chi² test passed, i.e. OE  retrieval agrees with
          Prior and null hypothesis is NOT rejected.
        chi2: real
          chi² value
        chi2TestY : real
          chi²  cutoff value with significance 'significance'

        """

        assert self.converged

        delta_y = self.y_i[self.convI] - self.y_a
        Sa = self.S_a.values
        S_ep = self.S_ep_i[self.convI]
        K = self.K_i[self.convI].values

        # Rodgers eq.12.16
        KSaK = K.dot(Sa).dot(K.T)
        KSaKSep_inv = invertMatrix(KSaK + S_ep)
        Syd = KSaK.dot(KSaKSep_inv).dot(KSaK)

        chi, chi2TestY = _testChi2(Syd, delta_y, significance, atol)

        #######  Alternative based on execise Rodgers 12.1 #######

        # Se = y_cov.values
        # K = self.K_i[self.convI].values
        # Sa = x_cov.sel(season=season).to_pandas().loc[x_vars,x_vars].values
        # d_y = (self.y_op[y_vars] - self.y_a[y_vars]).values

        # SaSqr = scipy.linalg.sqrtm(Sa)
        # SaSqr_inv = pyOE.pyOEcore.invertMatrix(SaSqr)

        # SeSqr = scipy.linalg.sqrtm(Se)
        # SeSqr_inv = pyOE.pyOEcore.invertMatrix(SeSqr)

        # Ktilde = SeSqr_inv.dot(K).dot(SaSqr)
        # U,s,vT = np.linalg.svd(Ktilde, full_matrices=False)
        # Lam = np.diag(s)
        # LamSq = Lam.dot(Lam)

        # m = len(y_vars)
        # invM= pyOE.pyOEcore.invertMatrix(LamSq + np.eye(m))
        # Sy = SeSqr.dot(U).dot(LamSq).dot(invM).dot(LamSq).dot(U.T).dot(SeSqr)

        # Sz4y = LamSq.dot(invM).dot(LamSq)
        # z4y = U.T.dot(SeSqr_inv).dot(d_y)

        # eigenvalues_compl = np.diag(Sz4y) # because it is diagonal

        # eigenvalues = s**4/(1+s**2) #equivalent!
        # assert np.isclose(eigenvalues_compl, eigenvalues).all()

        # notNull = ~np.isclose(0,eigenvalues)
        # chi2 = z4y[notNull]**2/eigenvalues[notNull]
        # chi2critical = scipy.stats.chi2.isf(significance, 1)

        return chi, chi2TestY

    def chiSquareTestXOptimalPrior(self, significance=0.05, atol=1e-5):
        """
        test with significance level 'significance' whether retrieval agrees
        with prior in x space (see chapter 12.3.3.3 of Rodgers, 2000)

        Parameters
        ----------
        significance  : real, optional
          significance level, defaults to 0.05, i.e. probability is 5% that
           correct null hypothesis is rejected.
        atol : float (default 1e-5)
            The absolute tolerance for comparing eigen values to zero. We 
            found that values should be than the numpy.isclose defualt value 
            of 1e-8.

        Returns
        -------
        chi2Passed : bool
          True if chi² test passed, i.e. OE  retrieval agrees with
          Prior and null hypothesis is NOT rejected.
        chi2 : real
          chi² value
        chi2TestX : real
          chi² cutoff value with significance 'significance'
        """

        assert self.converged

        delta_x = self.x_op - self.x_a

        Sa = self.S_a.values
        K = self.K_i[self.convI].values
        S_ep = self.S_ep_i[self.convI]

        # Rodgers eq. 12.12
        KSaKSep_inv = invertMatrix(K.dot(Sa).dot(K.T) + S_ep)
        Sxd = Sa.dot(K.T).dot(KSaKSep_inv).dot(K).dot(Sa)
        chi2, chi2TestX = _testChi2(Sxd, delta_x, significance, atol)

        #######  Alternative based on execise Rodgers 12.1 #######

        # Se = y_cov.values
        # K = self.K_i[self.convI].values
        # Sa = x_cov.sel(season=season).to_pandas().loc[x_vars,x_vars].values
        # d_x = (self.x_op[x_vars] - self.x_a[x_vars]).values

        # SaSqr = scipy.linalg.sqrtm(Sa)
        # SaSqr_inv = pyOE.pyOEcore.invertMatrix(SaSqr)

        # SeSqr = scipy.linalg.sqrtm(Se)
        # SeSqr_inv = pyOE.pyOEcore.invertMatrix(SeSqr)

        # Ktilde = SeSqr_inv.dot(K).dot(SaSqr)
        # U,s,vT = np.linalg.svd(Ktilde, full_matrices=False)
        # Lam = np.diag(s)

        # m = len(y_vars)
        # invM= pyOE.pyOEcore.invertMatrix(Lam.dot(Lam) + np.eye(m))
        # Sx = SaSqr.dot(vT.T).dot(Lam).dot(invM).dot(Lam).dot(vT)

        # z4x = vT.dot(SaSqr_inv).dot(d_x)
        # Sz4x = Lam.dot(invM).dot(Lam)

        # eigenvalues_compl = np.diag(Sz4x) # because it is diagonal
        # eigenvalues = s**2/(1+s**2) #equivalent!

        # assert np.isclose(eigenvalues_compl, eigenvalues).all()

        # notNull = ~np.isclose(0,eigenvalues)
        # chi2 = z4x[notNull]**2/eigenvalues[notNull]
        # chi2critical = scipy.stats.chi2.isf(significance, 1)

        return chi2, chi2TestX

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
        for k in list(oeDict.keys()):
            if k in ['forward', 'userJacobian']:
                oeDict.pop(k)

        np.save(fname, oeDict)
        return

    def plotIterations(
        self,
        cmap='viridis',
        figsize=(8, 10),
        legend=True,
        mode='ratio',
    ):
        r'''
        Plot the retrieval results using 4 panels: (1) iterations of x
        (normalized to self.x_truth or x[0]), (2) iterations of y (normalized
        to y_obs), (3) iterations of degrees of freedom, (4) iterations of
        convergence criteria

        Parameters
        ----------
        fileName : str, optional
          plot is saved to fileName, if provided
        cmap : str, optional
          colormap for 1st and 2nd panel (default 'hsv')
        figsize : tuple, optional
          Figure size in inch (default (8, 10))
        legend : bool, optional
          Add legend for X and Y (defualt True)
        mode : str, optional
          plot 'ratio' or 'difference' to truth/prior/measurements 
          (defualt: ratio)

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
            if mode == 'ratio':
                if self.x_truth is not None:
                    xs.append(self.x_truth[key])
                    xs = np.array(xs) / self.x_truth[key]
                else:
                    xs = np.array(xs) / xs[0]
            elif mode == 'difference':
                if self.x_truth is not None:
                    xs.append(self.x_truth[key])
                    xs = np.array(xs) - self.x_truth[key]
                else:
                    xs = np.array(xs) - xs[0]
            else:
                ValueError('Do not understand mode %s' % mode)
            sp1.plot(xs, label=key, color=colors[kk])
        if legend:
            leg = sp1.legend(loc="best",
                             prop=font_manager.FontProperties(size=8))
            leg.get_frame().set_alpha(0.5)
        # sp1.set_xlabel("iteration")
        if self.x_truth is not None:
            sp1.set_ylabel("x-values\n(%s to truth)" % mode)
        else:
            sp1.set_ylabel("x-values\n(%s to prior)" % mode)

        sp1.axvline(ind, color="k")
        sp1.axvline(len(self.x_i)-2, ls=":", color="k")

        colors = _niceColors(len(self.y_i[0].keys()), cmap=cmap)
        for kk, key in enumerate(self.y_i[0].keys()):
            ys = list()
            for yy in self.y_i:
                ys.append(yy[key])
            ys.append(self.y_obs[key])
            if mode == 'ratio':
                ys = np.array(ys) / ys[-1]
            elif mode == 'difference':
                ys = np.array(ys) - ys[-1]
            sp2.plot(ys, label=key, color=colors[kk])
        if legend:
            leg = sp2.legend(loc="best",
                             prop=font_manager.FontProperties(size=8))
            leg.get_frame().set_alpha(0.5)
        sp2.set_ylabel("y-values\n(%s to measurements)" % mode)
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

    def summary(self, *args, **kwargs):
        DeprecationWarning('Use summarize instead of summary!')
        return self.summarize(self, *args, **kwargs)

    def summarize(self, returnXarray=False, combineXB=False):
        '''Provide a summary of the retrieval results as a dictionary.

        Parameters
        ----------
        returnXarray : {bool}, optional
          return xarray dataset instead of dict. Can be easily combined when
          applying the retrieval multiple times. (the default is False)
        combineXB : {bool}, optional
          append b parameter values to state vector X variables. Can be useful
          for comparing runs with and without b parameters.

        Returns
        -------
        dict or xarray.Dataset
          Summary of retrieval results
        '''

        if self.convI < 0:
            raise RuntimeError("Retrieval did not run successfully")

        summary = {}
        summary['x_a'] = self.x_a.rename_axis('x_vars')
        summary['x_a_err'] = self.x_a_err.rename_axis('x_vars')
        summary['S_a'] = self.S_a.rename_axis(
            'x_vars').rename_axis('x_vars_T', axis=1)
        summary['x_op'] = self.x_op.rename_axis('x_vars')
        summary['x_op_err'] = self.x_op_err.rename_axis('x_vars')
        summary['S_op'] = self.S_op.rename_axis(
            'x_vars').rename_axis('x_vars_T', axis=1)
        summary['dgf_x'] = self.dgf_x.rename_axis('x_vars')
        summary['y_obs'] = self.y_obs.rename_axis('y_vars')
        summary['S_y'] = self.S_y.rename_axis(
            'y_vars').rename_axis('y_vars_T', axis=1)

        summary['y_op'] = self.y_op.rename_axis('y_vars')
        if self.x_truth is not None:
            summary['x_truth'] = self.x_truth.rename_axis('x_vars')

        if hasattr(self, 'nonlinearity'):
            summary['nonlinearity'] = self.linearity
        if hasattr(self, 'trueLinearityChi2'):
            summary['trueLinearityChi2'] = self.trueLinearityChi2
            summary['trueLinearityChi2Critical'] = \
                self.trueLinearityChi2Critical
        if hasattr(self, 'chi2Results'):
            summary['chi2value'] = self.chi2Results['chi2value']
            summary['chi2critical'] = self.chi2Results['chi2critical']

        summary['dgf'] = self.dgf_i[self.convI]
        summary['convergedIteration'] = self.convI

        if (not combineXB) and (len(self.b_vars) > 0):
            summary['b_p'] = self.b_p.rename_axis('b_vars')
            summary['S_b'] = self.S_b.rename_axis(
                'b_vars').rename_axis('b_vars_T', axis=1)
            summary['b_p_err'] = self.b_p_err.rename_axis('b_vars')

        elif combineXB and (len(self.b_vars) > 0):
            summary['x_a'] = pd.concat(
                (summary['x_a'], self.b_p)).rename_axis('x_vars')
            summary['x_op'] = pd.concat(
                (summary['x_op'], self.b_p)).rename_axis('x_vars')
            summary['x_op_err'] = pd.concat(
                (summary['x_op_err'], self.b_p_err)).rename_axis('x_vars')
            summary['dgf_x'] = pd.concat(
                (
                    summary['dgf_x'],
                    pd.Series(np.zeros(self.b_n), index=self.b_vars)
                )
            ).rename_axis('x_vars')
            summary['S_a'] = pd.concat(
                (summary['S_a'], self.S_b), sort=False
            ).rename_axis('x_vars').rename_axis('x_vars_T', axis=1)
            summary['S_op'] = pd.concat(
                (summary['S_op'], self.S_b), sort=False
            ).rename_axis('x_vars').rename_axis('x_vars_T', axis=1)

        if returnXarray:
            import xarray as xr
            summary = xr.Dataset(summary)

        return summary


def optimalEstimation_loadResults(fname, allow_pickle=True):
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
    oeDict = np.load(fname, allow_pickle=allow_pickle)
    oe = _oeDict2Object(oeDict.tolist())
    return oe


def invertMatrix(A, raise_error=True):
    '''
    Wrapper funtion for np.linalg.inv, because original function reports
    LinAlgError if nan in array for some numpy versions. We want that the
    retrieval is robust with respect to that. Also, checks for singular 
    matrices were added.

    Parameters
    ----------
    A : (..., M, M) array_like
        Matrix to be inverted.
    raise_error : {bool}, optional
        ValueError is raised if A is singular (the default is True)

    Returns
    -------
    Ainv : (..., M, M) ndarray or matrix
        Inverse of the matrix `A`.
    '''

    A = np.asarray(A)

    if np.any(np.isnan(A)):
        warnings.warn("Found nan in Matrix during inversion", UserWarning)
        return np.zeros_like(A) * np.nan

    try:
        eps = np.finfo(A.dtype).eps
    except:
        A = A.astype(np.float)
        eps = np.finfo(A.dtype).eps

    if np.linalg.cond(A) > 1/eps:
        if raise_error:
            raise ValueError("Found singular matrix", UserWarning)
        else:
            warnings.warn("Found singular matrix", UserWarning)
            return np.zeros_like(A) * np.nan
    else:
        return np.linalg.inv(A)


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
        oeDict.pop("y_obs"),
        oeDict.pop("S_y"),
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


def _estimateChi2(S, z, atol=1e-5):
    '''Estimate Chi^2 to estimate whether z is from distribution with 
    covariance S

    Parameters
    ----------
    S : {array}
        Covariance matrix
    z : {array}
        Vector to test
        atol : float (default 1e-5)
            The absolute tolerance for comparing eigen values to zero. We 
            found that values should be than the numpy.isclose defualt value 
            of 1e-8.

    Returns
    -------
    float
        Estimated chi2 value
    '''

    eigVals, eigVecsL = scipy.linalg.eig(S, left=True, right=False)
    z_prime = eigVecsL.T.dot(z)

    # Handle singular matrices. See Rodgers ch 12.2
    notNull = np.abs(eigVals) > atol
    dofs = np.sum(notNull)
    if dofs != len(notNull):
        print('Warning. Singular Matrix with rank %i instead of %i. '
              '(This is typically save to ignore)       ' %
              (dofs, len(notNull)))

    # Rodgers eq. 12.1
    chi2s = z_prime[notNull]**2/eigVals[notNull]
    return chi2s, dofs


def _testChi2(S, z, significance, atol=1e-5):
    '''Test whether z is from distribution with covariance S with significance

    Parameters
    ----------
    S : {array}
        Covariance matrix
    z : {array}
        Vector to test
    significance : {float}
        Significane level
        atol : float (default 1e-5)
            The absolute tolerance for comparing eigen values to zero. We 
            found that values should be than the numpy.isclose defualt value 
            of 1e-8.

    Returns
    -------
    float
        Estimated chi2 value
    float
        Theoretical chi2 value for significance
    bool
        True if Chi^2 test passed

    '''
    chi2s_obs, dof = _estimateChi2(S, z, atol=atol)
    chi2_obs = np.real_if_close(np.sum(chi2s_obs))
    chi2_theo = scipy.stats.chi2.isf(significance, dof)
    # chi2_theo1 = scipy.stats.chi2.isf(significance, 1)

    # print(chi2_obs<= chi2_theo, np.all(chi2s_obs<= chi2_theo1))

    return chi2_obs, chi2_theo
