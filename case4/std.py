import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import argparse
import datetime
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
from matplotlib.dates import DateFormatter
import matplotlib
from datetime import timedelta
sys.path.append(os.path.join(os.path.dirname(__file__), '../covid19/epidemics/cantons/py'))
from run_osp_cases import *
from nested_plot import *

ic_cantons=12
d1 = 24    

####################################################################################################
def getPosteriorFromResult(result):
####################################################################################################	
    from dynesty import utils as dyfunc
    weights = np.exp(result.logwt - result.logz[-1]) #normalized weights
    samples = dyfunc.resample_equal(result.samples, weights) #Compute 10%-90% quantiles.
    return samples
####################################################################################################	
def model(days,THETA,case):
####################################################################################################	
  ic_cantons = 12
  if case == 1:
    par = [THETA[0],THETA[1],THETA[2],THETA[3],THETA[4],THETA[5],  
           1000,1000,1000,
           1000,1000,1000]
    for i in range (6,6+ic_cantons):
        par.append(THETA[i])
    return example_run_seiin(days,par)
  elif case == 2:
    par = [THETA[0],THETA[1],THETA[2],THETA[3],THETA[4],THETA[5],  
           THETA[6],1000,1000,
           #THETA[6],THETA[7],1000,
           #d1,d2,1000]
           d1,1000,1000]
    #for i in range (8,8+ic_cantons):
    for i in range (7,7+ic_cantons):
        par.append(THETA[i])
    return example_run_seiin(days,par)


####################################################################################################
def standard_deviation(result,m=1):
####################################################################################################	
    name = ['AG','AI','AR','BE','BL','BS','FR','GE','GL','GR',\
            'JU','LU','NE','NW','OW','SG','SH','SO','SZ','TG',\
            'TI','UR','VD','VS','ZG','ZH']
    cantons = 26
    ic_cantons = 12
    days = 21 + 15
    if case == 2:
       days = 38 + 15

    samples = getPosteriorFromResult(result)
      
    prediction_matrix = np.zeros ( ( samples.shape[0]//m, days, 26 ) )  
    for i in range(  samples.shape[0]//m):
       print (i,"/",samples.shape[0]//m,flush=True)
       pp = samples[i,:]
       simulation = model(days,pp,case)
       prediction = []
       for d in range ( days ):
           cases = simulation[d].Iu()
           prediction_matrix   [i,d] = np.asarray(cases)

    return np.std(np.sum(prediction_matrix,axis=2),axis=0)








####################################################################################################
if __name__ == "__main__":

  argv = sys.argv[1:]
  parser = argparse.ArgumentParser()
  parser.add_argument('--case'  ,type=int)
  args = parser.parse_args(argv)

  m=1000
  case = args.case

  opt = []
  sub = []
  for sensors in range(1,11):
      res = pickle.load(open("./pickle_files/optimal_case"+str(case)+"_sensor"+str(sensors)+".pickle", "rb" ))
      opt.append(standard_deviation(res,m))
      res = pickle.load(open("./pickle_files/uniform_case"+str(case)+"_sensor"+str(sensors)+".pickle", "rb" ))
      sub.append(standard_deviation(res,m))
      
      days = len(sub[-1])
      #plt.plot(np.arange(days),opt[-1],label='opt'+str(sensors))
      #plt.plot(np.arange(days),sub[-1],label='sub'+str(sensors))
  


  
  days = len(sub[-1])
  max_opt = np.zeros(10)
  max_sub = np.zeros(10)
  for sensors in range(10):
     max_opt[sensors]  = np.max(opt[sensors])
     max_sub[sensors]  = np.max(sub[sensors])

  plt.plot(np.arange(10),max_opt,label='opt')
  plt.plot(np.arange(10),max_sub,label='sub')
  plt.legend()
  plt.show()
