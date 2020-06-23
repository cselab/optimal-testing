#!/usr/bin/env python3
import numpy as np
import pickle
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../covid19/epidemics/cantons/py'))
from run_osp_cases import *
import random

from mpi4py import MPI
import h5py


def getPosteriorFromResult(result):
    from dynesty import utils as dyfunc
    weights = np.exp(result.logwt - result.logz[-1]) #normalized weights
    samples = dyfunc.resample_equal(result.samples, weights) #Compute 10%-90% quantiles.
    return samples


def Posterior_Samples(days,samples,res):
    np.random.seed(1234567)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    size_1 = int ( np.sqrt(size) )
    size_2 = int ( np.sqrt(size) )
    if rank >= size_1 * size_2:
        print ("Rank",rank,"returns.")
        return
    N = size_1
    rank_1 = rank // N
    rank_2 = rank %  N


    s = getPosteriorFromResult(res)
    npar = 24 #parameters for seiir model    
    ic_cantons = 12 #nuisance parameters

    numbers = random.sample(range(s.shape[0]), samples)
    

    P     = np.zeros((ic_cantons,samples))
    j = 0
    for ID in numbers:
        P[0:ic_cantons+0,j] = s[ID,  s.shape[1] - ic_cantons - 1 : s.shape[1] - 1 ]
        j += 1

    params = s.shape[1] - 1 # -1 for the dispersion
    THETA = np.zeros((npar - ic_cantons,samples))
    ####################################
    if params == 6 + ic_cantons: #case 2
    ####################################
        j = 0
        for ID in numbers:
            u = np.random.uniform(0.0, 1.0, 1)
            THETA[0:6 ,j] = s[ID,  0:6 ] #(b0,mu,alpha,Z,D,theta)
            
            THETA[6   ,j] = s[ID,0]*u  #b1
            THETA[7   ,j] = s[ID,0]*u  #b2
            #mle = 0.90418851
            #THETA[6   ,j] =  u*mle #b1
            #THETA[7   ,j] =  u*mle #b2 

            THETA[8   ,j] = 10000.0    #b3
            THETA[9   ,j] = 21.0       #d1
            THETA[10  ,j] = 21.0       #d2
            THETA[11  ,j] = 10000.0    #d3       

            j += 1
    ####################################
    elif params == 6 + ic_cantons + 4: #case 3
    ####################################
        j = 0
        for ID in numbers:
            max_slope = 0.5
            u = np.random.uniform(0.0, 1.0, 1)
            THETA[0:8 ,j] = s[ID,  0:8 ] #(b0,mu,alpha,Z,D,theta,b1,b2)          
            THETA[8  ,j] =  u * max_slope
            THETA[9  ,j] = s[ID,  8] #d1
            THETA[10 ,j] = s[ID,  9] #d2
            THETA[11 ,j] = 84 #days      #d3
            j += 1

    All_results  = np.zeros((int(days),samples//N,samples//N,26))
    print (rank , All_results.shape)

    iii = 0
    for isim1 in range(rank_1*samples//N,(rank_1+1)*samples//N):
      for isim2 in range(rank_2*samples//N,(rank_2+1)*samples//N):
        p = []


        for i in range(npar- ic_cantons):
          p.append(THETA[i,isim1])
        for i in range(ic_cantons):
          p.append(P[i,isim2])

        results = example_run_seiin(days,p)    
        for day in range(days):
            All_results[int(day)][isim1-rank_1*samples//N][isim2-rank_2*samples//N][:] =  results[day].Iu()

        '''
        if isim1 == 0 and isim2 == 0:
           print("Params = ",p)
           prediction = []
           for day in range(days):
              prediction.append(results[day].Ir())
              print ("day=",day,results[day].Ir())
           prediction = np.asarray(prediction)
           np.save("prediction.npy",prediction)
           a = 1/0
        '''


        iii += 1


    print ("Rank",rank,"completed evaluations",flush=True)
    comm.Barrier()
  
    s = "{:05d}".format(samples)    
    name = "tensor_Ntheta=" + s
    f = h5py.File(name + ".hdf5", 'w', driver='mpio', comm=MPI.COMM_WORLD)
    dset = f.create_dataset('test', (int(days),samples,samples,26), dtype='f')
    dset[:,rank_1*samples//N:(rank_1+1)*samples//N,rank_2*samples//N:(rank_2+1)*samples//N,:] = All_results
    f.close()


def Uniform_Samples(days,samples):
    np.random.seed(1234567)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    size_1 = int ( np.sqrt(size) )
    size_2 = int ( np.sqrt(size) )
    if rank >= size_1 * size_2:
        print ("Rank",rank,"returns.")
        return
    N = size_1

    rank_1 = rank // N
    rank_2 = rank %  N

    npar = 24 #parameters for seiir model
    #            b     mu   a    Z    D  theta    b1   b2   b3    d1  d2   d3
    theta_min = [0.8, 0.2, 0.02, 1.0, 1.0, 1.0,   1000,1000,1000,1000,1000,1000]
    theta_max = [1.8, 1.0, 1.00, 6.0, 6.0, 1.4,   1000,1000,1000,1000,1000,1000]
    p_min = []
    p_max = []
    for i in range (12):
          p_min.append(0.0)
          p_max.append(30.)
    p_min = np.asarray(p_min)
    p_max = np.asarray(p_max)
    theta_min = np.asarray(theta_min)
    theta_max = np.asarray(theta_max)

    params = len(theta_min)
    THETA = np.random.uniform( 0.0, 1.0, (params,samples))
    for s in range(samples):
           THETA [:,s] = theta_min + (theta_max-theta_min)*THETA[:,s]
    P = np.random.uniform( 0.0, 1.0, (npar-params,samples))
    for s in range(samples):
           P [:,s] = p_min + (p_max-p_min)*P[:,s]
  
    

    All_results  = np.zeros((int(days),samples//N,samples//N,26))
    print (rank , All_results.shape)

    iii = 0
    for isim1 in range(rank_1*samples//N,(rank_1+1)*samples//N):
      for isim2 in range(rank_2*samples//N,(rank_2+1)*samples//N):
        p = []
        for i in range(params):
          p.append(THETA[i,isim1])
        for i in range(npar-params):
          p.append(P[i,isim2])

        results = example_run_seiin(days,p)    
        for day in range(days):
            All_results[int(day)][isim1-rank_1*samples//N][isim2-rank_2*samples//N][:] =  results[day].Iu()

        iii += 1


    print ("Rank",rank,"completed evaluations",flush=True)
    comm.Barrier()
  
    s = "{:05d}".format(samples)    
    name = "tensor_Ntheta=" + s
    f = h5py.File(name + ".hdf5", 'w', driver='mpio', comm=MPI.COMM_WORLD)
    dset = f.create_dataset('test', (int(days),samples,samples,26), dtype='f')
    dset[:,rank_1*samples//N:(rank_1+1)*samples//N,rank_2*samples//N:(rank_2+1)*samples//N,:] = All_results
    f.close()


if __name__ == '__main__':
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--days'       , type=int, default=22 )
    parser.add_argument('--samples'    , type=int, default=100)
    args = parser.parse_args(argv)
    model   = example_run_seiin

    '''
    Uniform_Samples    (args.days,args.samples)
    '''
    #res = pickle.load( open("case2/cantons___.pickle", "rb" ) )   
    res = pickle.load( open("case3/cantons___.pickle", "rb" ) )   
    Posterior_Samples(args.days,args.samples,res)
    



    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    filename = "tensor_Ntheta={:05d}.hdf5".format(args.samples)
    f = h5py.File(filename, 'r', driver='mpio', comm=comm)
    a_group_key = list(f.keys())[0]
    data = f[a_group_key]      

    if rank == 0:
        for i in range(args.days):
          s = "{:05d}".format(i)    
          name = "tensor_Ntheta=" + s
          print(i)
          d=data[i,:,:,:]
          np.save("case3/" + name + ".npy",d)
    f.close()
