import numpy as np
import pickle
import os
import sys
import argparse
import scipy
from scipy.special import loggamma
from dynesty import NestedSampler
from multiprocessing import Pool
import random
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../covid19/epidemics/cantons/py'))
from run_osp_cases import *
from scipy.stats import multivariate_normal
ic_cantons=12

def distance(t1,t2,tau):
    dt = np.abs(t1-t2) / tau
    return np.exp(-dt)

def GetNewSamples(path,day_max,case):

    np.random.seed(666)
    sigma_mean = np.zeros(day_max)  
    for i in range(day_max):
      temp = np.load(path + "tensor_Ntheta={:05d}.npy".format(i))
      sigma_mean[i] = np.mean(temp.flatten())

    #model run with maximum a posteriori estimate
    p_ = []    
    if case == 1:
      p = np.load("files/mle1.npy")
      p_ = [p[0],p[1],p[2],p[3],p[4],p[5]]
      for i in range(ic_cantons):
         p_.append(p[6+i])
    elif case == 2:
      p = np.load("files/mle2.npy")
      for i in range(ic_cantons+12):
         p_.append(p[i])
    r = example_run_seiin(day_max,p_,1000)


    Iu_all = np.zeros((day_max,26))

    c_real = 0.1
    tau_real = 2.0
    time   = np.arange(day_max)
    space  = np.zeros(day_max)
    aux = np.zeros((day_max,day_max))
    for i in range(day_max):
        for j in range(day_max):
              t1 = time [i]
              t2 = time [j] 
              s1 = space[i] 
              s2 = space[j] 
              if s1 == s2:
                 coef = distance(t1,t2,tau_real)  
                 #Small hack. When coef --> 1, two measurements are correlated and should not be both made
                 #If coef is not explicitly set to 1.0, we get covariance matrices that are ill-conditioned (det(cov)--> 0)
                 #and the results are weird. This hack guarantees numerical stability by explicitly making the covariance
                 #exactly singular.
                 if coef > 0.99: 
                    coef = 1.0
                 aux[i,j] = (sigma_mean[i]*sigma_mean[j])*coef
              else:
                 aux[i,j] = 0.0 
    COVARIANCE = (c_real*c_real)*aux


    for c1 in range(26):
        print ("Create data:",c1,flush=True)
        mean = np.zeros(day_max)
        for d1 in range(day_max):
            mean[d1] = r[d1].Iu()[c1]
        rv = scipy.stats.multivariate_normal(mean=mean, cov=COVARIANCE, allow_singular=True)
        Iu_all[:,c1] = rv.rvs(size=1)

    Iu_all[np.where(Iu_all<0)] = 0.0
    np.save("data_base.npy",Iu_all)


if __name__=='__main__':

    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--path'  ,type=str)
    parser.add_argument('--case'  ,type=int)
    args = parser.parse_args(argv)

    days = 21
    if args.case == 2:
      days = 60
    GetNewSamples(path=args.path,day_max=days,case=args.case)