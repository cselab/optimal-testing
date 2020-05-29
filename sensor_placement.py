#!/usr/bin/env python3
import argparse
from mpi4py import MPI
from osp import *

parser = argparse.ArgumentParser()
parser.add_argument('--nSensors', help='number of sensors to place'                                      , required=True, type=int)
parser.add_argument('--nMeasure', help='how many numbers describe a measurement taken by a single sensor', required=True, type=int)
parser.add_argument('--path'    , help='path to files to perform OSP'                                    , required=True, type=str)
parser.add_argument('--nProcs'  , help='number of processes to evaluate the utility'                     , required=True, type=int)
parser.add_argument('--Ny'      , help='number of MC samples for the Y integral'                         , required=True, type=int)
parser.add_argument('--Ntheta'  , help='number of MC samples for the THETA integral'                     , required=True, type=int)
parser.add_argument('--start_day', required=True, type=int)
args = vars(parser.parse_args())

import korali
k = korali.Engine()
k["Conduit"]["Type"] = "Distributed"
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

e_list = []
cantons = 26
for i in range (cantons):
  osp = OSP(path=args['path'],nSensors=args['nSensors'],nMeasure =args['nMeasure'],Ny=args['Ny'],Ntheta=args['Ntheta'],korali=1,start_day=args['start_day'])
  osp.current_canton = i
  days = osp.data.shape[1]

  e = korali.Experiment()
  e["Random Seed"] = 0xC0FEE
  e["Problem"]["Type"]               = "Optimization/Stochastic"
  e["Problem"]["Objective Function"] = osp.EvaluateUtility2
  e["Variables"][0]["Name"] = "dummy"
  e["Solver"]["Type"]                   = "SequentialOptimisation"
  e["Solver"]["Number Of Sensors"]      = args['nSensors']
  e["Solver"]["Locations Per Variable"] = [ days ]
  e["Console Output"]["Verbosity"] = "Detailed"
  #e["Console Output"]["Verbosity"] = "Silent"
  e_list.append(e)

k.run(e_list)

  
if rank == size -1:
   F = np.zeros((args['nSensors'],cantons,days))
   for i in range (cantons):
       utility = np.array(e_list[i]["Solver"]["Utility"])
       F[:,i,:]= utility
       L    = np.zeros(args['nSensors'])
       Fmax = np.zeros(args['nSensors'])
       for n in range(args['nSensors']):
           L[n] = np.argmax(F[n,i,:])
           Fmax[n] = np.max(F[n,i,:])
       print("Canton",i," LOCATION:", L  ,"  OBJECTIVE:",Fmax)

   np.save("result_Ny{:05d}_Nt{:05d}.npy".format(osp.Ny,osp.Ntheta),F)
