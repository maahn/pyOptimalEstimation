"pyOptimalEstimation" Package
*****************************

Python package to solve an inverse problem using Optimal Estimation
and an arbritrary Forward model following Rodgers, 2000.


"pyOEcore" Module
=================

class class pyOptimalEstimation.pyOEcore.optimalEstimation(x_vars, x_ap, x_cov, y_vars, y_cov, y_obs, forward, x_lowerLimit={}, x_upperLimit={}, useFactorInJac=True, gammaFactor=None, disturbance=1.01, convergenceFactor=10, forwardKwArgs={})

   Bases: "object"

   The core optimalEstimation class, which contains all required
   parameters.

   Parameters :
      **x_vars** : list of str

         names of the elements of state vector x.

      **x_ap** : pn.Series or list or np.ndarray

         prior information of state x.

      **x_cov** : pn.DataFrame or list or np.ndarray

         covariance matrix of state x.

      **y_vars** : list of str

         names of the elements of state vector x

      **y_cov** : pn.DataFrame or list or np.ndarray

         covariance matrix of measurement y.

      **y_obs** : pn.Series or list or np.ndarray

         observed measurement vector y.

      **forward** : function

         forward model expected as "forward(x,**forwardKwArgs): return
         y"

      **forwardKwArgs** : dict,optional

         additional keyword arguments for forward function

      **x_lowerLimit** : dict, optional

         reset state vector x[key] to x_lowerLimit[key] in case
         x_lowerLimit is undercut. defaults to {}.

      **x_upperLimitn** : dict, optional

         reset state vector x[key] to x_upperLimit[key] in case
         x_upperLimit is exceeded. defaults to {}.

      **disturbance** : float or dict of floats, optional

         relative disturbance of statet vector x to estimate the
         Jacobian. Can be specified for every element of x seperately.
         Defaults to 1.01.

      **useFactorInJac** : bool,optional

         True if disturbance should be applied by multiplication,
         False if it should by applied by additiion. Defaults to True.

      **gammaFactor** : list of floats, optional

         Use additional gamma parameter for retrieval [R1].

   Returns :
      pyOptimalEstimation object

         returns the pyOptimalEstimation object

   -[ References ]-

   [R1] Turner, D. D., and U. Löhnert, 2014: Information Content and
        Uncertainties in Thermodynamic Profiles and Liquid Cloud
        Properties Retrieved from the Ground-Based Atmospheric Emitted
        Radiance Interferometer (AERI). Journal of Applied Meteorology
        & Climatology, 53, 752–771, doi:10.1175/JAMC-D-13-0126.1.

   -[ Attributes ]-

   +----------------+------------------------------------------------------------------------------------------+
   | converged      | (boolean) True if retriveal converged successfully                                       |
   +----------------+------------------------------------------------------------------------------------------+
   | K_i            | (list of pn.DataFrame) list of Jacobians for iteration i.                                |
   +----------------+------------------------------------------------------------------------------------------+
   | x_i            | (list of pn.Series) iterations of state vector x                                         |
   +----------------+------------------------------------------------------------------------------------------+
   | y_i            | (list of pn.Series) iterations of measurement vector y                                   |
   +----------------+------------------------------------------------------------------------------------------+
   | dgf_i          | (list of float) degrees of freedom for each iteration                                    |
   +----------------+------------------------------------------------------------------------------------------+
   | A_i            | (list of pn.DataFrame) Averaging kernel for each iteration                               |
   +----------------+------------------------------------------------------------------------------------------+
   | d_i2           | (list of float) convergence criteria for each iteration                                  |
   +----------------+------------------------------------------------------------------------------------------+
   | S_aposterior_i | (list of pn.DataFrame) a posteriori covariance matrix of x for each iteration            |
   +----------------+------------------------------------------------------------------------------------------+
   | gam_i          | (list of floats) gamma parameters used in retrievals, see also *gammaFactor* and  [R1].  |
   +----------------+------------------------------------------------------------------------------------------+

   -[ Methods ]-

   doRetrieval(maxIter=10)

      run the retrieval

      Parameters :
         **maxIter** : int, optional

            maximum number of iterations, defaults to 10

      Returns :
         bool

            True is convergence was obtained.

   getJacobian(x)

      estimate Jacobian using the forward model and the specified
      disturbance

      Parameters :
         **x** : pn.Series or list or np.ndarray

            state vector x

      Returns :
         pn.DataFrame

            Jacobian around x

   plotIterations(fileName=None, x_truth=None)

      Plot the retrieval results using 4 panels: (1) iterations of x
      (normalized to x_truth or x[0]), (2) iterations of y (normalized
      to y_obs), (3)  iterations of degrees of freedom, (4) iterations
      of convergence criteria

      Parameters :
         **fileName** : str, optional

            plot is saved to fileName, if provided

         **x_truth** : pn.Series or list or np.ndarray, optional

            If truth of state x is known, it can be included in the
            plot.

      Returns :
         matplotlib figure object

            The created figure.

   saveResults(fname)

      Helper function to save a pyOptimalEstimation object. The
      forward operator is removed from the pyOptimalEstimation object
      before saving.

      Parameters :
         **fname** : str

            filename

      Returns :
         None

pyOptimalEstimation.pyOEcore.optimalEstimation_loadResults(fname)

   Helper function to load a saved pyOptimalEstimation object

   Parameters :
      **fname** : str

         filename

   Returns :
      pyOptimalEstimation object

         pyOptimalEstimation obtained from file.
