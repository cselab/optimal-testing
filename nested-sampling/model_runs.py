#!/usr/bin/env python3
import numpy as np
import pickle
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../covid19/epidemics/cantons/py'))
from run_osp_cases import *
import random


def getPosteriorFromResult(result):
    from dynesty import utils as dyfunc
    weights = np.exp(result.logwt - result.logz[-1]) #normalized weights
    samples = dyfunc.resample_equal(result.samples, weights) #Compute 10%-90% quantiles.
    return samples


def Posterior_Samples(days,samples,res):
    np.random.seed(1234567)

    s = getPosteriorFromResult(res)
    ic_cantons = 12 #nuisance parameters

    numbers = random.sample(range(s.shape[0]), samples)

    P     = np.zeros((ic_cantons,samples))
    j = 0
    for ID in numbers:
        P[0:ic_cantons+0,j] = s[ID,  s.shape[1] - ic_cantons - 1 : s.shape[1] - 1 ]
        j += 1

    params = s.shape[1] - 1 # -1 for the dispersion
    THETA = []
    assert params == 12 + ic_cantons #case 3

    dispersion = np.zeros(samples)
    i = 0
    for ID in numbers:
        THETA.append(s[ID,0:12]) #(b0,mu,alpha,Z,D,theta,b1,b2,d1,d2,theta1,theta2)
        dispersion[i] = s[ID,-1]
        i += 1

    All_results  = np.zeros((int(days),samples,samples,26))

    iii = 0
    for isim1 in range(samples):
      print (isim1)
      for isim2 in range(samples):
        p = []
        for i in range(len(THETA[0])):
          p.append(THETA[isim1][i])
        for i in range(ic_cantons):
          p.append(P[i,isim2])
        results = example_run_seiin(days,p)

        aux = p[2]/p[3]
        for day in range(days):
            All_results[int(day),isim1,isim2,:] =  aux*np.asarray(results[day].E())
            #All_results[int(day),isim1,isim2,:] =  results[day].Iu()
        iii += 1

    np.save("runs.npy",All_results)
    np.save("dispersion.npy",dispersion)



if __name__ == '__main__':

      model   = example_run_seiin
      samples = 100
      days = 160
      res = pickle.load( open("case3/cantons___3.pickle", "rb" ) )
      Posterior_Samples(days,samples,res)
