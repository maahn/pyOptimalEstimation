# -*- coding: utf-8 -*-
#pyOptimalEstimation

#Copyright (C) 2014-15 Maximilian Maahn, IGMK (mmaahn_(AT)_meteo.uni-koeln.de)
#http://gop.meteo.uni-koeln.de/software


import cPickle as pickle
import time
from copy import deepcopy 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager 
import pandas as pn
import warnings
  
class optimalEstimation(object):
  r'''
  The core optimalEstimation class, which contains all required parameters.

  Parameters
  ---------- 
  x_vars : list of str
      names of the elements of state vector x.
  x_ap : pn.Series or list or np.ndarray
      prior information of state x.
  x_cov : pn.DataFrame or list or np.ndarray
      covariance matrix of state x.
  y_vars : list of str
      names of the elements of state vector x
  y_cov : pn.DataFrame or list or np.ndarray
      covariance matrix of measurement y.
  y_obs : pn.Series or list or np.ndarray
      observed measurement vector y.
  forward : function
      forward model expected as ``forward(x,**forwardKwArgs): return y``
  forwardKwArgs : dict,optional
      additional keyword arguments for forward function
  x_lowerLimit : dict, optional
      reset state vector x[key] to x_lowerLimit[key] in case x_lowerLimit is undercut. defaults to {}.
  x_upperLimitn : dict, optional
      reset state vector x[key] to x_upperLimit[key] in case x_upperLimit is exceeded. defaults to {}.
  disturbance : float or dict of floats, optional
      relative disturbance of statet vector x to estimate the Jacobian. Can be specified for every element of x seperately. Defaults to 1.01.
  useFactorInJac : bool,optional
      True if disturbance should be applied by multiplication, False if it should by applied by additiion. Defaults to True.
  gammaFactor : list of floats, optional
      Use additional gamma parameter for retrieval [1]_. 

  Attributes
  ----------
  converged : boolean
    True if retriveal converged successfully
  K_i : list of pn.DataFrame
    list of Jacobians for iteration i.
  x_i : list of pn.Series
    iterations of state vector x
  y_i : list of pn.Series
    iterations of measurement vector y
  dgf_i : list of float 
    degrees of freedom for each iteration
  A_i  : list of pn.DataFrame
    Averaging kernel for each iteration
  d_i2 : list of float 
    convergence criteria for each iteration
  S_aposterior_i : list of pn.DataFrame
    a posteriori covariance matrix of x for each iteration
  gam_i : list of floats
    gamma parameters used in retrievals, see also `gammaFactor` and  [1]_. 
    
  Returns
  -------
  
  pyOptimalEstimation object
    returns the pyOptimalEstimation object 


  References
  ----------
  .. [1] Turner, D. D., and U. Löhnert, 2014: Information Content and Uncertainties in Thermodynamic Profiles and Liquid Cloud Properties Retrieved from the Ground-Based Atmospheric Emitted Radiance Interferometer (AERI). Journal of Applied Meteorology & Climatology, 53, 752–771, doi:10.1175/JAMC-D-13-0126.1.

  '''  
  def __init__(self, x_vars, x_ap, x_cov, y_vars, y_cov, y_obs, forward, x_lowerLimit={}, x_upperLimit={}, useFactorInJac = True, gammaFactor=None, disturbance =1.01, convergenceFactor=10, forwardKwArgs = {}):
    self.x_vars = x_vars
    self.x_ap = pn.Series(x_ap,index=self.x_vars)
    self.x_cov = pn.DataFrame(x_cov,index=self.x_vars,columns=self.x_vars)
    self.x_n = len(self.x_vars)
    self.y_vars = y_vars 
    self.y_cov = pn.DataFrame(y_cov,index=self.y_vars,columns=self.y_vars)
    self.y_obs = pn.Series(y_obs,index=self.y_vars)
    self.y_n = len(self.y_vars)
    self.forward = forward
    try: self.forward_name = forward.__name__  #We want to save at least the name because the forward function is removed for saving
    except AttributeError:  self.forward_name = None
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
    self.dgf_i=None
    self.A_i = None
    self.d_i2 =None
    self.S_aposterior_i = None
    self.Pxy_i = None
    self.gam_i = None
    
    
  def getJacobian(self,x):
    r'''
    estimate Jacobian using the forward model and the specified disturbance

    Parameters
    ---------- 
    x  : pn.Series or list or np.ndarray
      state vector x

    Returns
    -------
    pn.DataFrame
      Jacobian around x
    '''  
    x = pn.Series(x,index=self.x_vars,dtype=float)
    if type(self.disturbance) == float:
      disturbances = dict()
      for x_key in self.x_vars: disturbances[x_key] = self.disturbance
    elif type(self.disturbance) == dict:
       disturbances = self.disturbance
    else:
      raise TypeError("disturbance must be type dict or float")
    
    disturbedKeys = ["reference"]
    for tup in self.x_vars:
      disturbedKeys.append("disturbed %s"%tup)
    self.x_disturbed = pn.DataFrame(columns=self.x_vars, index=disturbedKeys,dtype=float)
    self.x_disturbed.ix["reference"] = x
    for x_key in self.x_vars:
      disturbed_x_key = "disturbed %s"%x_key
      self.x_disturbed.ix[disturbed_x_key] = x
      #apply disturbance here!!
      if self.useFactorInJac: self.x_disturbed[x_key][disturbed_x_key] = x[x_key] * disturbances[x_key]
      else: self.x_disturbed[x_key][disturbed_x_key] = x[x_key] + disturbances[x_key]
      #import pdb;pdb.set_trace()
    self.y_disturbed = pn.DataFrame(columns=self.y_vars, index=disturbedKeys)
    for x_dist in self.x_disturbed.index:
       self.y_disturbed.ix[x_dist] = self.forward(self.x_disturbed.ix[x_dist],**self.forwardKwArgs)

    y = self.y_disturbed.ix["reference"]
    
    #remove the reference from the disturbed keys!
    disturbedKeys = disturbedKeys[1:]
    #create an empty jacobian matrix
    jacobian = pn.DataFrame(np.ones((self.y_n,self.x_n)),index=self.y_vars,columns=disturbedKeys)
    #calc Jacobian
    for y_key in self.y_vars:
      for x_key in self.x_vars:
        #realtive disturbance
        #import pdb;pdb.set_trace()
        if self.useFactorInJac: dist = x[x_key] * (disturbances[x_key] -1)
        else: dist = disturbances[x_key]
        jacobian["disturbed "+x_key][y_key] = (self.y_disturbed[y_key]["disturbed "+x_key] - y[y_key]) /dist
    
    jacobian[np.isnan(jacobian)+np.isinf(jacobian)] = 0.
    return jacobian




  def doRetrieval(self, maxIter=10, maxTime = 1e7):
    r"""
    run the retrieval
    
    Parameters
    ---------- 
    maxIter  : int, optional
      maximum number of iterations, defaults to 10
    maxTime  : int, optional
      maximum runTime, defaults to 1e7 (~ 4 months).
      Note that the forward model is *not* killed if time is exceeded

    Returns
    -------
    bool
      True is convergence was obtained.
    
    """
    
    
    assert maxIter >0
    self.converged = False
    startTime = time.time()
    
    S_a = np.array(self.x_cov) #Covariance of prior estimate of x
    S_Ep = np.array(self.y_cov) #S_Epsilon Covariance of measurement noise
    S_Ep_inv = _invertMatrix(S_Ep) #S_Ep inverted
    S_a_inv = _invertMatrix(S_a) #S_a inverted
    self.K_i = [0]*maxIter #list of jacobians
    self.x_i = [0]*(maxIter+1)
    self.y_i = [0]*maxIter
    self.dgf_i=[0]*maxIter
    self.A_i = [0]*maxIter
    self.d_i2 =[0]*maxIter #convergence criteria
    self.S_aposterior_i = [0] *maxIter
    self.Pxy_i = [0] *maxIter
    self.gam_i = [1]*maxIter
    if self.gammaFactor:
      assert len(self.gammaFactor) <= maxIter    
      self.gam_i[:len(self.gammaFactor)] = self.gammaFactor

    self.x_i[0] = self.x_ap
    self.d_i2[0] = 1e333
    
    for i in range(maxIter):
      self.K_i[i] = self.getJacobian(self.x_i[i])
    
      assert np.all(self.y_disturbed.keys() == self.y_cov.keys())
      assert np.all(self.y_cov.keys() == self.K_i[i].index)
      assert np.all(self.x_cov.index ==  self.x_ap.index)
      assert np.all( self.x_ap.index ==  self.x_disturbed.columns)
      assert np.all(self.x_disturbed.index[1:] == self.K_i[i].columns)
      
      self.y_i[i] = self.y_disturbed.ix["reference"]
      K = np.array(self.K_i[i])
      
      #reformulated using Turner and Löhnert 2013:
      B = (self.gam_i[i] * S_a_inv) + K.T.dot(S_Ep_inv.dot(K)) #eq 3
      B_inv = _invertMatrix(B) 
      self.S_aposterior_i[i] = B_inv.dot((self.gam_i[i]**2 * S_a_inv) + K.T.dot(S_Ep_inv.dot(K))).dot(B_inv) # eq2
      self.S_aposterior_i[i] = pn.DataFrame(self.S_aposterior_i[i],index=self.x_ap.index,columns=self.x_ap.index)
      G = B_inv.dot(K.T.dot(S_Ep_inv))
      self.A_i[i] = G.dot(K) #eq 4
      #import pdb;pdb.set_trace()
      self.x_i[i+1] = self.x_ap + _invertMatrix((self.gam_i[i] * S_a_inv) + K.T.dot(S_Ep_inv.dot(K))).dot(K.T.dot(S_Ep_inv.dot(self.y_obs - self.y_i[i] + K.dot(self.x_i[i]-self.x_ap)))) #eq 1
      #import pdb;pdb.set_trace()
      self.dgf_i[i] = np.trace(self.A_i[i])
      
      #check whether i+1 is valid
      for jj,xKey in enumerate(self.x_vars):
        if xKey in self.x_lowerLimit.keys() and self.x_i[i+1][jj] < self.x_lowerLimit[xKey]: 
          print "#"*60
          print "reset due to x_lowerLimit: %s from %f to %f in iteration %d"%(xKey,self.x_i[i+1][jj],self.x_ap[jj],i)
          self.x_i[i+1][jj] = self.x_ap[jj]
        if xKey in self.x_upperLimit.keys() and self.x_i[i+1][jj] > self.x_upperLimit[xKey]: 
          print "#"*60
          print "reset due to x_upperLimit: %s from %f to %f in iteration %d"%(xKey,self.x_i[i+1][jj],self.x_ap[jj],i)
          self.x_i[i+1][jj] = self.x_ap[jj]        
        if np.isnan(self.x_i[i+1][jj]):
          print "#"*60
          print "reset due to nan: %s from %f to %f in iteration %d"%(xKey,self.x_i[i+1][jj],self.x_ap[jj],i)
          self.x_i[i+1][jj] = self.x_ap[jj]        

          
      #convergence criteria
      self.d_i2[i] = (self.x_i[i] - self.x_i[i+1]).T.dot(_invertMatrix(self.S_aposterior_i[i])).dot(self.x_i[i] - self.x_i[i+1]) #eq 6
      
      #stop if we converged in the step before
      if self.converged:
        print time.time()-startTime, "s, iteration", i, "degrees of freedom:", self.dgf_i[i], "of", self.x_n,  "Done.", self.d_i2[i]
        break
      
      elif ((time.time()-startTime)> maxTime):
          print time.time()-startTime, "s, iteration", i, "DoF:", self.dgf_i[i], "of", self.x_n, "maximum Time exceeded! STOP", self.d_i2[i]
          self.converged = False
          #failed = True
          break
            
      #calculate the convergence criteria
      if i!=0:
        if np.abs(self.d_i2[i]) < self.y_n/float(self.convergenceFactor) and self.gam_i[i] == 1 and self.d_i2[i] != 0:
          print time.time()-startTime, "s, iteration", i, "degrees of freedom:", self.dgf_i[i], "of", self.x_n,  "convergence criteria fullfilled", self.d_i2[i]
          self.converged = True
        elif (i>1) and (self.dgf_i[i] == 0):
          print time.time()-startTime, "s, iteration", i, "DoF:", self.dgf_i[i], "of", self.x_n, "degrees of freedom 0! STOP", self.d_i2[i]
          self.converged = False
          #failed = True
          break
        else:
          print time.time()-startTime, "s, iteration", i, "DoF:", self.dgf_i[i], "of", self.x_n,   "convergence criteria NOT fullfilled", self.d_i2[i]
    self.K_i = self.K_i[:i+1]
    self.x_i = self.x_i[:i+2]
    self.y_i = self.y_i[:i+1]
    self.dgf_i = self.dgf_i[:i+1]
    self.A_i = self.A_i[:i+1]
    self.d_i2 = self.d_i2[:i+1]
    self.S_aposterior_i = self.S_aposterior_i[:i+1]
    self.Pxy_i = self.Pxy_i[:i+1]
    self.gam_i = self.gam_i[:i+1]
    if self.converged:
      self.convI = i
    else:
      self.convI = -9999
    return self.converged
  
  def saveResults(self,fname):
    r'''
    Helper function to save a pyOptimalEstimation object. The forward operator
    is removed from the pyOptimalEstimation object before saving. 

    Parameters
    ----------
    fname : str
      filename

    Returns
    -------
    None
    '''
    oeDict = deepcopy(self.__dict__)
    if "forward" in oeDict.keys(): oeDict.pop("forward")
    np.save(fname,oeDict)
    return
  
  def plotIterations(self,fileName=None,x_truth=None):
    r'''
    Plot the retrieval results using 4 panels: (1) iterations of x (normalized
    to x_truth or x[0]), (2) iterations of y (normalized to y_obs), (3) 
    iterations of degrees of freedom, (4) iterations of convergence criteria

    Parameters
    ----------
    fileName : str, optional
      plot is saved to fileName, if provided
    x_truth : pn.Series or list or np.ndarray, optional
      If truth of state x is known, it can be included in the plot.

    Returns
    -------
    matplotlib figure object
      The created figure.
    '''    
    fig = plt.figure(figsize=(8,10))
    d_i2 = np.array(self.d_i2)
    dgf_i = np.array(self.dgf_i)
    
    try:
      gamma = np.array(self.gam_i)
      noGam = len(gamma[gamma != 1])
      ind = np.argmin(d_i2[noGam:]) +noGam - 1
    except: ind = 0
    
    if self.converged: fig.suptitle('converged ' + str(d_i2[ind])+ " "+ str(dgf_i[ind]))
    else: fig.suptitle('NOT converged ' + str(d_i2[ind]) + " "+ str(dgf_i[ind]))
    sp = fig.add_subplot(411)
    colors = _niceColors(len(self.x_i[0].keys()))
    for kk,key in enumerate(self.x_i[0].keys()):
      xs = list()
      for xx in self.x_i[:-1]:
        xs.append(xx[key])
      if x_truth is not None: 
        xs.append(x_truth[key])
        xs = np.array(xs) / x_truth[key]
      else:
        xs = np.array(xs) / xs[0]
      sp.plot(xs,label=key,color=colors[kk])
    l = sp.legend(loc="upper left",prop=font_manager.FontProperties(size=8))
    l.get_frame().set_alpha(0.5)
    #sp.set_xlabel("iteration")
    sp.set_ylabel("normalized y-value")
    #sp.yaxis.set_label_coords(-0.08, 0.5)
    sp.axvline(ind,color="k")
    sp.set_xlim(0,len(self.x_i)-1)
    sp.axvline(len(self.x_i)-2,ls=":",color="k")
    sp.set_xticklabels("")
    sp = fig.add_subplot(412)
    colors = _niceColors(len(self.y_i[0].keys()))        
    for kk,key in enumerate(self.y_i[0].keys()):
      ys = list()
      for yy in self.y_i: ys.append(yy[key])
      ys.append(self.y_obs[key])
      ys = np.array(ys)/ ys[-1]
      sp.plot(ys,label=key,color=colors[kk])
    l = sp.legend(loc="upper left",prop=font_manager.FontProperties(size=8))
    l.get_frame().set_alpha(0.5)
    #sp.set_xlabel("iteration")
    sp.set_ylabel("normalized x-value")
    #sp.yaxis.set_label_coords(-0.08, 0.5)
    sp.axvline(ind,color="k")
    sp.axvline(len(self.x_i)-2,ls=":",color="k")
    sp.set_xlim(0,len(self.x_i)-1)
    sp.set_xticklabels("")
    sp = fig.add_subplot(413)
    sp.plot(dgf_i,label="degrees of freedom")
    #sp.yaxis.set_label_coords(-0.08, 0.5)
    #sp.set_xlabel("iteration")
    sp.set_ylabel("dgf")
    sp.axvline(len(self.x_i)-2,ls=":",color="k")
    sp.axvline(ind,color="k")
    sp.set_xlim(0,len(self.x_i)-1)
    sp.set_xticklabels("")
    sp = fig.add_subplot(414)
    sp.plot(d_i2,label="d_i2")
    sp.set_xlabel("iteration")
    sp.set_ylabel("convergence criterion")
    #sp.yaxis.set_label_coords(-0.08, 0.5)
    fig.subplots_adjust(hspace=0.1)
    sp.set_xlim(0,len(self.x_i)-1)
    sp.axvline(len(self.x_i)-2,ls=":",color="k")
    sp.axvline(ind,color="k")
    xlabels = map(lambda x:"%i"%x,sp.get_xticks())
    xlabels[-1] = "truth"
    sp.set_xticklabels(xlabels)
    if fileName:
      fig.savefig(fileName)
    return fig


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
  Helper function to convert a oe-dictionary (usually load from a file) to a pyOptimalEstimation object

  Parameters
  ----------
  oeDict : dict
    dictionary object

  Returns
  -------
  pyOptimalEstimation object
    pyOptimalEstimation obtained from file.
  '''
  oe = optimalEstimation(oeDict.pop("x_vars"), oeDict.pop("x_ap"), oeDict.pop("x_cov"), oeDict.pop("y_vars"), oeDict.pop("y_cov"), oeDict.pop("y_obs"), None)
  for kk in oeDict.keys():
    oe.__dict__[kk] = oeDict[kk]
  return oe  
  
  
def _niceColors(length,cmap='hsv'):
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
  Wrapper funtion for np.linalg.inv, because original function reports LinAlgError if nan in array for some numpy versions. We want taht the retrieval is robust with respect to that
  '''
  if np.any(np.isnan(A)):
    warnings.warn("Found nan in Matrix during inversion",UserWarning)
    return np.zeros_like(A) * np.nan
  else:
    return np.linalg.inv(A) 

