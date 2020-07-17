#!/usr/bin/env python3
import argparse,sys
from mpi4py import MPI
from osp import *
from common import *

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

start_day = T_S_CASE_1
days      = T_E_CASE_1
if args['case'] == 2:
   start_day = T_S_CASE_2
   days      = T_E_CASE_2
if args['case'] == 3:
   start_day = T_S_CASE_3
   days      = T_E_CASE_3
if args['case'] == 4: #case 3b
   start_day = T_S_CASE_4
   days      = T_E_CASE_4



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
e["Solver"]["Locations Per Variable"] = [ days , CANTONS ]
k["Conduit"]["Type"] = "Distributed"

# Running Korali
k.run(e)

if rank == size-1:
  utility = e["Solver"]["Utility"]
  utility = np.array(utility)
  utility = utility.reshape((args['nSensors'],CANTONS,days))
  if case == 1:   
     np.save("result_Ny{:05d}_Nt{:05d}_1.npy".format(osp.Ny,osp.Ntheta),utility)
  if case == 2:   
     np.save("result_Ny{:05d}_Nt{:05d}_2.npy".format(osp.Ny,osp.Ntheta),utility)
  if case == 3:   
     np.save("result_Ny{:05d}_Nt{:05d}_3.npy".format(osp.Ny,osp.Ntheta),utility)
  if case == 4:   
     np.save("result_Ny{:05d}_Nt{:05d}_4.npy".format(osp.Ny,osp.Ntheta),utility)
