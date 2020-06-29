#!/usr/bin/env python3

# Importing computational model
import argparse
import sys
from mpi4py import MPI
sys.path.append('../covid19/applications/osp/')
from osp import *
parser = argparse.ArgumentParser()
parser.add_argument('--nSensors',help='number of sensors to place'       , required=True, type=int)
parser.add_argument('--path'    ,help='path to files to perform OSP'     , required=True, type=str)
parser.add_argument('--Ny'      ,help='MC samples for the Y integral'    , required=True, type=int)
parser.add_argument('--Ntheta'  ,help='MC samples for the THETA integral', required=True, type=int)
parser.add_argument('--case'    ,help='case 1,2 or 3'                    , required=True, type=int)
args = vars(parser.parse_args())


case  = args['case']
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if rank ==0:
   print("Running case : ", case)

NUM_CANTONS = 26
start_day = 0
days      = 8
if args['case'] == 2:
   start_day = 21
   days      = 21 + 14
if args['case'] == 3:
   start_day = 117
   days      = 117 + 40

osp = OSP(path=args['path'],nSensors=args['nSensors'],Ny=args['Ny'],Ntheta=args['Ntheta'],start_day=start_day,days=days,korali=1)

# Starting Korali's Engine
import korali
k = korali.Engine()

# Creating new experiment
e = korali.Experiment()

# Configuring Problem
e["Random Seed"] = 0xC0FEE
e["Problem"]["Type"] = "Optimization/Stochastic"
e["Problem"]["Objective Function"] = osp.EvaluateUtility

e["Variables"][0]["Name"] = "Dummy"

# Configuring sequential optimisation parameters
e["Solver"]["Type"] = "SequentialOptimisation"
e["Solver"]["Number Of Sensors"] = args['nSensors'] 
e["Solver"]["Locations Per Variable"] = [ days , NUM_CANTONS ]
#e["Console Output"]["Verbosity"] = "Detailed"

k["Conduit"]["Type"] = "Distributed"
# Running Korali
k.run(e)

if rank == size-1:
  utility = e["Solver"]["Utility"]
  utility = np.array(utility)
  utility = utility.reshape((args['nSensors'],NUM_CANTONS,days))
  if case == 1:   
     np.save("result_Ny{:05d}_Nt{:05d}_1.npy".format(osp.Ny,osp.Ntheta),utility)
  if case == 2:   
     np.save("result_Ny{:05d}_Nt{:05d}_2.npy".format(osp.Ny,osp.Ntheta),utility)
  if case == 3:   
     np.save("result_Ny{:05d}_Nt{:05d}_3.npy".format(osp.Ny,osp.Ntheta),utility)
