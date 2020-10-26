#!/usr/bin/env python3
import argparse,sys
from mpi4py import MPI
from ospStandalone import *

NCANTONS = 26

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
    recvbuf = np.empty([size, nPerRank], dtype=np.float64)

  optimalTime = []
  optimalLocation = []
  utility = []
  for survey in range( nSurveys ):
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
      if idx > nFunctionEvaluations:
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
    utility = utility.reshape((nSurveys,NCANTONS,days)) 
    np.save("case{}/result_Ny{:05d}_Nt{:05d}.npy".format(case,osp.Ny,osp.Ntheta),utility)
