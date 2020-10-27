#!/usr/bin/env python3
import argparse,sys
from mpi4py import MPI
import numpy as np
import scipy.stats as sp

NCANTONS = 26

def distance(t1,t2,tau):
    dt = np.abs(t1-t2)/tau
    return np.exp(-dt)

class OSP:
  #########################################################################################################
  def __init__(self, path, nSensors = 1, Ntheta = 100, Ny = 100 , start_day = -1,days=-1):
  #########################################################################################################
    self.path       = path        # path to tensor_.npy
    self.nSensors   = nSensors    # how many sensors to place
    self.Ny         = Ny          # how many samples to draw when evaluating utility
    self.Ntheta     = Ntheta      # model parameters samples
    self.days       = days        # how many days (i.e. sensor locations)
    self.start_day  = start_day      
    self.sigma_mean = np.zeros(self.days)  

    self.sigma      = [0.05,0.10,0.15,0.20]
    self.tau        = [1.0,2.0,3.0]
    self.wtau       = 1.0/len(self.tau)
    self.wsigma     = 1.0/len(self.sigma)

    for i in range(self.days):
      temp = np.load(path+"/day={:05d}.npy".format(i))
      self.sigma_mean[i] = np.mean(temp.flatten())

  #############################################################################################
  def EvaluateUtility(self, argument):
  #############################################################################################
    space = []
    time  = []
    
    n = int ( len(argument)/2 )
    for i in range(n):
      space.append(argument[i])
      time.append(argument[i+n])

    for i in range(n-1):
      if time[n-1] == time[i] and space[n-1] == space[i]:
        st = []
        for j in range(n-1):
          st.append(space[j])
        for j in range(n-1):
          st.append(time[j])
        retval = self.EvaluateUtility(st)
        return retval


    Ntheta   = self.Ntheta
    Ny       = self.Ny
    F_tensor = np.zeros( (Ntheta, Ntheta, n ))
    for s in range(n):
      temp = np.load(self.path+"/day={:05d}.npy".format(time[s]))
      F_tensor[:,:,s] = temp[:,:,space[s]] 

    #Estimate covariance matrix as a function of the sensor locations (time and space)
    rv_list = []
    covariances = []
    sigma_mean = np.zeros(n)
    for i in range(n):
      sigma_mean[i] = self.sigma_mean[time[i]]
    for i_tau in range(len(self.tau)):
      aux = np.zeros((n,n))
      for i in range(n):
        for j in range(n):
            t1 = time [i]
            t2 = time [j] 
            s1 = space[i] 
            s2 = space[j] 
            if s1 == s2:
               coef = distance(t1,t2,self.tau[i_tau]) 
               #Small hack. When coef --> 1, two measurements are correlated and should not be both made
               #If coef is not explicitly set to 1.0, we get covariance matrices that are ill-conditioned (det(cov)--> 0)
               #and the results are weird. This hack guarantees numerical stability by explicitly making the covariance
               #exactly singular.
               if coef > 0.99:
                  coef = 1.0
               aux[i,j] = (sigma_mean[i]*sigma_mean[j])*coef
            else:
               aux[i,j] = 0.0 

      for i_sigma in range(len(self.sigma)):
        aux1 = self.sigma[i_sigma]**2 * aux
        rv_list.append(sp.multivariate_normal(np.zeros(n), aux1, allow_singular=True))
        covariances.append(aux1)

    #compute utility
    retval = 0.0
    for i_tau in range(len(self.tau)):
      for i_sigma in range(len(self.sigma)):
        jjj  = i_tau * len(self.sigma) + i_sigma
        aux  = covariances[jjj] 
        rv   = rv_list[jjj]

        for theta in range(Ntheta):

          mean = F_tensor[theta,theta,:]
          y    = np.random.multivariate_normal(mean=mean, cov=aux, size=Ny)  
          s1   = np.mean(rv.logpdf(y-mean))

          #this is a faster way to avoid a second for loop over Ntheta
          evidence = np.zeros((Ntheta,Ny))
          s2 = 0.0
          m1,m2 = np.meshgrid(y[:,0],F_tensor[:,theta,0] )
          new_shape1 = m1.shape[0]*m1.shape[1]
          new_shape2 = m2.shape[0]*m2.shape[1]
          m1 = m1.reshape((new_shape1,1))
          m2 = m2.reshape((new_shape2,1))
          for ns in range(1,n):
               m1_tmp,m2_tmp = np.meshgrid(y[:,ns],F_tensor[:,theta,ns] )
               m1_tmp = m1_tmp.reshape(m1_tmp.shape[0]*m1_tmp.shape[1],1)
               m2_tmp = m2_tmp.reshape(m2_tmp.shape[0]*m2_tmp.shape[1],1)
               m1=np.concatenate( (m1,m1_tmp), axis= 1 )
               m2=np.concatenate( (m2,m2_tmp), axis= 1 )
          for i_sigma1 in range(len(self.sigma)):
            jjj1  = i_tau * len(self.sigma) + i_sigma1
            evidence += (rv_list[i_sigma1].pdf(m1-m2)).reshape((Ntheta,Ny))

          s2 += np.mean ( np.log( self.wsigma*np.mean( evidence,axis=0) ) )
          retval += (s1-s2)
    retval *= self.wsigma/self.Ntheta 
    retval *= self.wtau
    return retval


if __name__ == '__main__':
  ### Read the command line arguments and run setup ###
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  parser = argparse.ArgumentParser()
  parser.add_argument('--nSurveys',help='number of surveys to perform'       , required=True, type=int)
  parser.add_argument('--path'    ,help='path to files with model evaluations for sampled parameters'     , required=True, type=str)
  parser.add_argument('--Ny'      ,help='MC samples for the Y integral'    , required=True, type=int)
  parser.add_argument('--Ntheta'  ,help='MC samples for the THETA integral', required=True, type=int)
  parser.add_argument('--case'    ,help='case 1,2 or 3'                    , required=True, type=int)
  args = vars(parser.parse_args())

  nSurveys = args['nSurveys']
  path     = args['path']
  Ny       = args['Ny']
  Ntheta   = args['Ntheta']
  case     = args['case']
  if rank ==0:
     print("#### Computing Utility Function ####")
     print("####################################")
     print("case",case)
     print("nSurveys",nSurveys)
     print("path", path)
     print("Ny", Ny)
     print("Ntheta",Ntheta)
     print("numRanks",size)
     print("####################################")

  # Setting ranges according to the selected case
  start_day = 0
  days      = 8
  if args['case'] == 2:
     start_day = 21
     days      = 35
  if args['case'] == 3:
     start_day = 102
     days      = 110
  if args['case'] == 4: #case 3b
     start_day = 136
     days      = 144

  osp = OSP(path=path, nSensors=nSurveys, Ny=Ny, Ntheta=Ntheta, start_day=start_day, days=days)

  ### Compute the Utility Function ####
  nFunctionEvaluations = NCANTONS*(days-start_day)
  nPerRank = (nFunctionEvaluations + size - 1) // size
  sendbuf = np.zeros(nPerRank,dtype=np.float64)
  recvbuf = None
  if rank == 0:
    print("Running {} function evaluations.. {} per Rank".format(nFunctionEvaluations,nPerRank))
  optimalTime = []
  optimalLocation = []
  utility = []
  for survey in range( nSurveys ):
    recvbuf = np.empty([size, nPerRank], dtype=np.float64)
    # create container for time and space indices
    time = np.zeros(survey+1,dtype=int)
    space = np.zeros(survey+1,dtype=int)
    # add previously found optima to end
    if survey > 0:
      time[-survey:] = optimalTime[:survey]
      space[-survey:] = optimalLocation[:survey]
    # evaluate utility for candidates on the different ranks
    for evaluation in range(nPerRank):
      idx = rank*nPerRank+evaluation
      if idx >= nFunctionEvaluations:
        break
      time[0] = start_day+idx%(days-start_day)
      space[0] = idx//days
      sendbuf[evaluation] = osp.EvaluateUtility( space.tolist() + time.tolist() )
    # gather the results from the rank
    comm.Gather(sendbuf, recvbuf, root=0)
    # add computed utility to container and compute the argmax
    if rank == 0:
      recvbuf = recvbuf.reshape((-1,))[:nFunctionEvaluations]
      utility.append(recvbuf)
      maxIdx = np.argmax( recvbuf )
      optimalTime.append( maxIdx%days )
      optimalLocation.append( maxIdx//days )
      print("Survey {}: Time {}, Location {}".format(survey,optimalTime[survey],optimalLocation[survey]))
    optimalTime = comm.bcast(optimalTime, root=0)
    optimalLocation = comm.bcast(optimalLocation, root=0)

  # write result to file
  if rank == 0:
    utility = np.array(utility)
    utility = utility.reshape((nSurveys,NCANTONS,(days-start_day))) 
    np.save("case{}/result_Ny{:05d}_Nt{:05d}.npy".format(case,osp.Ny,osp.Ntheta),utility)
