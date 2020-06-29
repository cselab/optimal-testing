#!/usr/bin/env python3
import numpy as np
import pickle
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../covid19/epidemics/cantons/py'))
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
    ic_cantons = 12 #nuisance parameters

    numbers = random.sample(range(s.shape[0]), samples)


    P     = np.zeros((ic_cantons,samples))
    j = 0
    for ID in numbers:
        P[0:ic_cantons+0,j] = s[ID,  s.shape[1] - ic_cantons - 1 : s.shape[1] - 1 ]
        j += 1

    params = s.shape[1] - 1 # -1 for the dispersion
    THETA = []
    ####################################
    if params == 6 + ic_cantons: #case 2
    ####################################
        for ID in numbers:
            THETA.append(s[ID,0:6]) #(b0,mu,alpha,Z,D,theta)
    ####################################
    elif params == 12 + ic_cantons: #case 3
    ####################################
        for ID in numbers:
            THETA.append(s[ID,0:12]) #(b0,mu,alpha,Z,D,theta,b1,b2,d1,d2,theta1,theta2)

    All_results  = np.zeros((int(days),samples//N,samples//N,26))

    print (rank , All_results.shape)
    iii = 0
    for isim1 in range(rank_1*samples//N,(rank_1+1)*samples//N):
      for isim2 in range(rank_2*samples//N,(rank_2+1)*samples//N):
        p = []
        for i in range(len(THETA[0])):
          p.append(THETA[isim1][i])
        for i in range(ic_cantons):
          p.append(P[i,isim2])
        results = example_run_seiin(days,p)
        for day in range(days):
            All_results[int(day),isim1-rank_1*samples//N,isim2-rank_2*samples//N,:] =  results[day].Iu()
        iii += 1
    print ("Rank",rank,"completed evaluations",flush=True)
    comm.Barrier()
    return All_results

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

     npar = 6 + 12 #parameters for seiir model
     #            b     mu   a    Z    D  theta
     theta_min = [0.8, 0.2, 0.02, 1.0, 1.0, 0.5]
     theta_max = [1.8, 1.0, 1.00, 6.0, 6.0, 1.5]
     p_min = []
     p_max = []
     for i in range (12):
           p_min.append(0.0)
           p_max.append(50.)
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
         results = example_run_seiin(days,p,1000)
         for day in range(days):
             All_results[int(day)][isim1-rank_1*samples//N][isim2-rank_2*samples//N][:] =  results[day].Iu()
         iii += 1
     print ("Rank",rank,"completed evaluations",flush=True)
     comm.Barrier()
     return All_results



if __name__ == '__main__':
      argv = sys.argv[1:]
      parser = argparse.ArgumentParser()
      parser.add_argument('--samples'    , type=int, default=100)
      parser.add_argument('--case'       , type=int, default=1)

      args = parser.parse_args(argv)
      model   = example_run_seiin
      samples = args.samples

      days = 0
      if args.case == 1:
         days = 30
      elif args.case == 2:
         days = 60
      elif args.case == 3:
         days = 160

      comm = MPI.COMM_WORLD
      rank = comm.Get_rank()
      size = comm.Get_size()
      size_1 = int ( np.sqrt(size) )
      size_2 = int ( np.sqrt(size) )
      N = size_1
      rank_1 = rank // N
      rank_2 = rank %  N
      results  = np.zeros((int(days),samples//N,samples//N,26))

      if args.case == 1:
         results = Uniform_Samples    (days,args.samples)
      elif args.case == 2:
         res = pickle.load( open("case2/cantons___2.pickle", "rb" ) )
         results = Posterior_Samples(days,args.samples,res)
      elif args.case == 3:
         res = pickle.load( open("case3/cantons___3.pickle", "rb" ) )
         results =  Posterior_Samples(days,args.samples,res)

      comm = MPI.COMM_WORLD
      rank = comm.Get_rank()
      for d in range(days):
           comm.Barrier()

           if rank == 0:
              data = np.zeros((samples,samples,26))
              data[0:samples//N,0:samples//N,:] = results[d,:,:,:]
              for r in range(1,size):
                rank_1 = r // N
                rank_2 = r %  N
                data[rank_1*samples//N:(rank_1+1)*samples//N,
                     rank_2*samples//N:(rank_2+1)*samples//N,:] = comm.recv(source=r, tag=r)
              s = "{:05d}".format(d)
              name = "tensor_Ntheta=" + s
              print("Saving day",d,"...",end='')
              np.save("case"+str(args.case)+"/" + name + ".npy",data)
              print("completed.")
              del data
           else:
              comm.send(results[d,:,:,:], dest=0, tag=rank)