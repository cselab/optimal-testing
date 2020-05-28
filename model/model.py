#!/usr/bin/env python

import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../covid19/'))

# Importing the classes for the computational model
import libepidemics.cantons.seiin_interventions as seiin_interventions
import libepidemics.cantons.seiin as seiin

from epidemics.cantons.py.model import \
        get_canton_model_data, get_canton_reference_data, \
        get_municipality_model_data, ModelData

from epidemics.data.swiss_cantons import CANTON_KEYS_ALPHABETICAL, CANTON_POPULATION

from epidemics.tools.autodiff import  cantons_custom_derivatives

"""wrapper to run SEIIR model with samples from Korali

    Arguments:
        sample: Korali sample object with parameters, IC and sigma.
        ntime: number of timesteps to run simulation.
        mTMCMC: Flag specifying whether use mTMCMC (need gradients or not).
        scenario: string specifying the scenario to run inference
    """


def setIC(sample, nParams, nIC) :
  
  y0 = np.zeros(5*26)

  # Get N and set S=N
  y0[-26:] = [ CANTON_POPULATION[c] for c in CANTON_KEYS_ALPHABETICAL]
  y0[:26]  = [ CANTON_POPULATION[c] for c in CANTON_KEYS_ALPHABETICAL]
  
  # samples for unreported cases in 12 cantons
  cantons = [0,3,4,5,6,7,9,15,20,22,23,25]
  
  for i in range(nIC):
    y0[3*26+cantons[i]] = sample["Parameters"][nParams+i]
    y0[cantons[i]] = y0[cantons[i]] - y0[3*26+cantons[i]]

  # one reported case in ticino
  y0[2*26+20] = 1
  y0[20] = y0[20] - 1
  return y0


def runCantonsSEIIN( sample, ntime, mTMCMC, scenario, x ):
  nIC = 12
  nParams = 6
  # Get model parameters from korali
  beta  = sample["Parameters"][0]
  mu    = sample["Parameters"][1]
  alpha = sample["Parameters"][2]
  Z     = sample["Parameters"][3]
  D     = sample["Parameters"][4]
  theta = sample["Parameters"][5]
  
  if scenario == "second-outbreak":
    nParams = 10
    b1    = sample["Parameters"][6]
    b2    = sample["Parameters"][7]
    d1    = sample["Parameters"][8]
    d2    = sample["Parameters"][9]

  # Create Epidemiological Model Class
  data = get_canton_model_data()
  
  y0 = setIC(sample, nParams, nIC)
  # print(y0)

  solver = []  
  params = []
  if scenario == "social-distancing":
    y0     = seiin.State(y0)
    solver = seiin.Solver(data.to_cpp())
    params = seiin.Parameters(beta=beta, mu=mu, alpha=alpha, Z=Z, D=D, theta=theta)

  elif scenario == "second-outbreak":
    y0     = seiin_interventions.State(y0)
    solver = seiin_interventions.Solver(data.to_cpp())
    params = seiin_interventions.Parameters(beta=beta, mu=mu, alpha=alpha, Z=Z, D=D, theta=theta, b1=b1, b2=b2, b3=0, d1=d1, d2=d2, d3=ntime)

  # Get standard deviation from Korali
  sigma = sample["Parameters"][-1]

  sdevgrad = np.zeros((nParams+nIC)*5)
  sdevgrad[-(nParams+nIC):] = 1
  sdevgrad = sdevgrad.tolist()

  # Run model for time interval specified
  t_eval = np.arange(ntime)
  results = []
  if mTMCMC:
    params_der = np.zeros((nParams, nParams+nIC))
    for p in range(nParams):
      params_der[p,p] = 1

    y0_der = np.zeros((5*26, nParams+nIC))
    for i,c in enumerate(cantons):
      y0_der[3*26+c,nParams+i] = 1

      static_ad = solver.solve_params_ad(params, y0, t_eval=t_eval, dt=0.1)
      results, der_results = cantons_custom_derivatives(solver, params, y0, params_der, y0_der, t_eval=t_eval, dt=0.1)

  else:
      results = solver.solve(params, y0, t_eval=t_eval, dt=0.1)

  # gather results and pass them to Korali
  daily_cases = [] 
  std = []
  cantons_inference = True
  if not mTMCMC:
    if cantons_inference == True:
      for i in range(len(x)):
        c = int(x[i] / 10000)
        d = x[i] - c*10000
        cases = results[d].E()
        daily_cases.append(alpha/Z *cases[c])
        std.append(sample["Parameters"][-1] * alpha/Z *cases[c])
    else:
      for i in range(len(x)):
        d = i
        cases = results[d].E() 
        daily_cases.append(alpha/Z *np.sum(cases) )
        std.append(s["Parameters"][-1] * alpha/Z *np.sum(cases))
  else:
    daily_cases = [] 
    std = []
    der_daily_cases = [] 
    der_std = []
    if cantons_inference == True:
      for i in range(len(x)):
        c = int(x[i] / 10000)
        d = x[i] - c*10000
        # get daily_cases
        daily_cases += [ alpha/Z*results[d,1,c] ]

        # get derivative of daily_cases
        der_cases = alpha/Z*der_results[d,1,c,:]
        # theta = a
        der_cases[2] += results[d,1,c]/Z
        # theta = Z
        der_cases[3] -= alpha*results[d,1,c]/Z**2
        # der_std
        der_cases = der_cases.tolist()

        # add std of derivative of daily cases
        der_std += der_cases
        der_std += [ alpha/Z*results[d,1,c] ]
        # add derivative for sigma
        der_cases += [ 0 ]
        # put derivatives to container
        der_daily_cases += der_cases
    else:
      print("Only cantons_inference available for mTMCMC")

    sample["Gradient Mean"]               = der_daily_cases
    sample["Gradient Standard Deviation"] = der_std

  sample["Reference Evaluations"] = daily_cases
  # ensure that sdv is not 0.0
  daily_cases = [x or 1e-10 for x in daily_cases]
  sample["Standard Deviation"] = (np.array(daily_cases)*sigma).tolist()

  # print("Passed results to Korali")


def getReferenceData( ntime ):
 cantons_inference = True
 if cantons_inference == False:
     data = np.load("canton_daily_cases.npy")
     cantons = data.shape[0] # = 26
     days    = data.shape[1]
     assert cantons == 26
     y = []
     for d in range (days):
      if d >= ntime:
        continue
      tot = 0
      for c in range (cantons):
        if np.isnan(data[c,d]) == False:
          tot += data[c,d]
      y.append(tot)
     return y
 else:
     data = np.load("canton_daily_cases.npy")
     cantons = data.shape[0] # = 26
     days    = data.shape[1]
     assert cantons == 26
     y = []
     for c in range(cantons):
       for d in range(days):
        if d >= ntime:
          continue
        if np.isnan(data[c,d]) == False:
          y.append(data[c,d])
     return y

def getReferencePoints( ntime ):
 cantons_inference = True
 if cantons_inference == False:
    x=[] 
    data = np.load("canton_daily_cases.npy")
    cantons = data.shape[0] # = 26
    days    = data.shape[1]
    assert cantons == 26
    x = []
    for d in range(days):
     if d >= ntime:
          continue
     x.append(d)
    return x
 else:
    x=[] 
    data = np.load("canton_daily_cases.npy")
    cantons = data.shape[0] # = 26
    days    = data.shape[1]
    assert cantons == 26
    x = []
    for c in range(cantons):
      for d in range(days):
        if d >= ntime:
          continue
        if np.isnan(data[c,d]) == False:
            x.append(d + c*10000)
    return x
