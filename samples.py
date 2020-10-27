#!/usr/bin/env python3
import numpy as np
import pickle,argparse,sys,os,random
from mpi4py import MPI
from seiin import *

def getPosteriorFromResult(result):
    from dynesty import utils as dyfunc
    weights = np.exp(result.logwt - result.logz[-1]) #normalized weights
    samples = dyfunc.resample_equal(result.samples, weights) #Compute 10%-90% quantiles.
    return samples

def Posterior_Samples(days,samples,res,case):
    np.random.seed(1234567)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

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
    dispersion = np.zeros(samples)
    i = 0

    if params == 6 + ic_cantons: #case 2
        for ID in numbers:
            THETA.append(s[ID,0:6]) #(b0,mu,alpha,Z,D,theta)
    elif params == 12 + ic_cantons: #case 3
        for ID in numbers:
            THETA.append(s[ID,0:12]) #(b0,mu,alpha,Z,D,theta,b1,b2,d1,d2,theta1,theta2)
            dispersion[i] = s[ID,-1]
            i += 1
    elif params == 12 + ic_cantons + 1: #case 4
        for ID in numbers:
            THETA.append(s[ID,0:13]) #(b0,mu,alpha,Z,D,theta,b1,b2,d1,d2,theta1,theta2,lambda)
            dispersion[i] = s[ID,-1]
            i += 1

    All_results  = np.zeros((int(days),samples//size,samples,26))

    iii = 0
    for isim1 in range(rank*samples//size,(rank+1)*samples//size):
      for isim2 in range(samples):
        p = []
        for i in range(len(THETA[0])):
          p.append(THETA[isim1][i])
        for i in range(ic_cantons):
          p.append(P[i,isim2])
        results = example_run_seiin(days,p)
        aux = p[2]/p[3]
        for day in range(days):
            All_results[int(day),isim1-rank*samples//size,isim2,:] =  aux* np.asarray(results[day].E())
        iii += 1
    comm.Barrier()

    np.save("case"+str(case)+"/dispersion.npy",dispersion)
    np.save("case"+str(case)+"/runs.npy",All_results)

    return All_results

def Uniform_Samples(days,samples):
     np.random.seed(1234567)
     comm = MPI.COMM_WORLD
     rank = comm.Get_rank()
     size = comm.Get_size()

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

     All_results  = np.zeros((int(days),samples//size,samples,26))
     iii = 0
     for isim1 in range(rank*samples//size,(rank+1)*samples//size):
       for isim2 in range(samples):
         p = []
         for i in range(params):
           p.append(THETA[i,isim1])
         for i in range(npar-params):
           p.append(P[i,isim2])
         results = example_run_seiin(days,p,1000)
         for day in range(days):
             All_results[int(day)][isim1-rank*samples//size][isim2][:] =  results[day].Iu()
         iii += 1
     comm.Barrier()
     return All_results

if __name__ == '__main__':
      argv = sys.argv[1:]
      parser = argparse.ArgumentParser()
      parser.add_argument('--samples', type=int, default=100)
      parser.add_argument('--case'   , type=int, default=1)

      args = parser.parse_args(argv)
      model   = example_run_seiin
      samples = args.samples
      case = args.case

      days = 0
      if case == 1:
         days = 20
      elif case == 2:
         days = 60
      elif case == 3:
         days = 120
      elif case == 4:
         days = 150

      print("+++++++++++++++++++++++++++++++++++++++++++++++")
      print("++ Model evaluations using parameter samples ++")
      print(" Case: ", case)
      print(" Samples: ", samples)
      print("+++++++++++++++++++++++++++++++++++++++++++++++")
      
      comm = MPI.COMM_WORLD
      rank = comm.Get_rank()
      size = comm.Get_size()

      assert samples % size == 0

      results  = np.zeros((int(days),samples//size,samples,26))

      from pathlib import Path
      Path("case"+str(case)).mkdir(parents=True, exist_ok=True)

      if case == 1:
         results = Uniform_Samples(days,samples)
      else:
         res = pickle.load( open("case"+str(case)+"/samples_"+str(case)+".pickle", "rb" ) )
         results = Posterior_Samples(days,samples,res,case)

      comm = MPI.COMM_WORLD
      rank = comm.Get_rank()
      for d in range(days):
           comm.Barrier()
           if rank == 0:
              data = np.zeros((samples,samples,26))
              data[0:samples//size, :, :] = results[d,:,:,:]
              for r in range(1,size):
                data[r*samples//size:(r+1)*samples//size, :,:] = comm.recv(source=r, tag=r)
              s = "{:05d}".format(d)
              name = "day=" + s
              np.save("case"+str(case)+"/" + name + ".npy",data)
              del data
           else:
              comm.send(results[d,:,:,:], dest=0, tag=rank)
