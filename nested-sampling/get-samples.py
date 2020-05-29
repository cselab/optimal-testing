import numpy as np
import pickle
import argparse
import sys
import os
#sys.path.append(os.path.join(os.path.dirname(__file__), '../covid19/epidemics/cantons/py'))
from epidemics.cantons.py.run_osp_cases import *
import random

def getPosteriorFromResult(result):
    from dynesty import utils as dyfunc
    weights = np.exp(result.logwt - result.logz[-1]) #normalized weights
    samples = dyfunc.resample_equal(result.samples, weights) #Compute 10%-90% quantiles.
    return samples

def Non_Uniform_Samples(days,samples):
    basename = "cantons___"
    res = pickle.load( open( basename + ".pickle", "rb" ) )
    parameters = getPosteriorFromResult(res)
    l = len(parameters)
    numbers = random.sample(range(l), samples)

    out = np.zeros((samples,days,26))
    i = 0

    data = np.load("canton_daily_cases.npy")
    print(data.shape)
    

    for ID in numbers:
        print (i,"/",samples)
        p = parameters[ID,:]

        if parameters.shape[1] == 19: #case 2: b0,mu,a,Z,D,theta, I_IC(12), dispersion
           print("CASE 2")

           P = np.random.uniform( 0.0, 1.0, 2)
           d0 = 21 #+ P[0] * (days-21)

           p_ = []
           p_.append(p[0])
           p_.append(p[1])
           p_.append(p[2])
           p_.append(p[3])
           p_.append(p[4])
           p_.append(p[5])
           p_.append(P[1]*p[0])
           p_.append(P[1]*p[0])
           p_.append(10000)
           p_.append(d0)
           p_.append(d0)
           p_.append(10000)
           for j in range(12):
               p_.append(p[6+j])

           simulation = example_run_seiin(days,p_)
           for d in range ( days ):
               out[i,d,:] = np.asarray( simulation[d].Iu() )
        else:
           print("CASE 3")
           assert parameters.shape[1] == 23
           P = np.random.uniform( 0.0, 1.0, 1)
 
           p_ = []
           p_.append(p[0])
           p_.append(p[1])
           p_.append(p[2])
           p_.append(p[3])
           p_.append(p[4])
           p_.append(p[5])

           p_.append(p[6])
           p_.append(p[7])
           #p_.append(p[0]*P)
           max_slope = 0.05
           p_.append( P * max_slope )


           p_.append(p[8])
           p_.append(p[9])
           p_.append(data.shape[1]) #=days
           for j in range(12):
               p_.append(p[10+j])
          

 
           simulation = example_run_seiin(days,p_)
           for d in range ( days ):
               out[i,d,:] = np.asarray( simulation[d].Iu() )
        i += 1
    #np.save("output_Ntheta={:05d}.npy".format(samples),out)
    np.save('output_Ntheta={:05d}.npy'.format(samples),out)


def Uniform_Samples(days,samples):
    npar = 24 #parameters for seiir model
    #        b0   mu   a    Z    D    theta   interventions (off)
    p_min = [0.2, 0.1, 0.0, 1.0, 1.0, 0.8   , 1000,1000,1000,1000,1000,1000 ]
    p_max = [3.5, 1.4, 1.0, 10., 10., 4.0   , 1000,1000,1000,1000,1000,1000 ]
    for i in range (12):
          p_min.append(0.0   )
          p_max.append(1000.0)
          #p_min.append(i*15)
          #p_max.append(i*15)
    p_min = np.asarray(p_min)
    p_max = np.asarray(p_max)
    P = np.random.uniform( 0.0, 1.0, (npar,samples))
    for s in range(samples):
           P [:,s] = p_min + (p_max-p_min)*P[:,s]
    
    All_results  = np.zeros((samples,int(days),26))
    all_params   = np.zeros((samples,npar))
    for isim in range(samples):
        #results = model(days,P[:,isim])
        results = example_run_seiin(days,P[:,isim])
        for day in range(0,days):
            All_results[isim][int(day)][:] = results[day].Iu()
        all_params[isim]=P[:,isim]
        print (isim + 1,"/",samples)
    np.save("output_Ntheta={:05d}.npy".format(samples),All_results)


if __name__ == '__main__':
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('--days'       , type=int, default=40  )
    parser.add_argument('--samples'    , type=int, default=100 )
    parser.add_argument('--uniform'    , type=int, default=1   )
    args = parser.parse_args(argv)

    model   = example_run_seiin

    if args.uniform == 1:
       Uniform_Samples    (args.days,args.samples)
    else:
       Non_Uniform_Samples(args.days,args.samples)
