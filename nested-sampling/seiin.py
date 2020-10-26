#!/usr/bin/env python3
'''
Wrapper for the SEIIN model with interventions from the COVID-19 library.
The necessary inputs differ for cases 1,2 and 3.

Case 1: start of an epidemic
Inputs: β,μ,α,Z,D,θ,I_{IC}

Case 2: first intervention announced 
Inputs: b_0,μ,α,Z,D,θ_0,I_{IC},int_day
        In this case, b_1 and θ_1 are picked uniformly at random.

Case 3: measures loosened 
Inputs: b_0,μ,α,Z,D,θ,I_{IC}, b_1,b_2,d_1,d_2,θ_1,θ_2
        In this case, b_3 is picked uniformly at random 

Parameters:
β : transmission rate when the epidemic starts
μ : unreported infections have a smaller transmission rate: β*μ, 0<μ<1
α : reporting rate (fraction of total infections that are reported)
Z : virus latency period (measured in days)
D : infectious period (measured in days)
θ : mobility factor when the epidemic starts
I_{IC}: vector with the initial conditions for the unreported infections in 12 out of 26 cantons
int_day : day of the 1st intervention (=21)
d_1     : day of the 1st intervention (same as int_day)
d_2     : day of the 2nd intervention 
b_1,θ_1 : transmission rate and mobility factor after d_1
b_2,θ_2 : transmission rate and mobility factor after d_2
b_3     : after loosening of measures, the transmission rate is given by min(b_0,b_3*(t-102))
'''
import numpy as np
import os,sys
sys.path.append('../covid19/')
sys.path.append('../')
from epidemics.model import get_canton_model_data
from common import *
import libepidemics

data = get_canton_model_data(include_foreign=False)

def example_run_seiin(num_days, inputs, int_day=T_S_CASE_2):
    num_days = int(num_days)
    cantons_ = [0,3,4,5,6,7,9,15,20,22,23,25]
    L = len(inputs)
    ic_cantons = len(cantons_)
    N0  = list(data.region_population)
    E0  = [0] * data.num_regions
    IR0 = [0] * data.num_regions
    IU0 = [0] * data.num_regions
    IR0[data.key_to_index['TI']] = 1  # Ticino.

    if L == 6 + ic_cantons: #inference/model evaluations for case 2 
       params = libepidemics.Parameters(
                 beta  =inputs[0],
                 mu    =inputs[1],
                 alpha =inputs[2],
                 Z     =inputs[3],
                 D     =inputs[4],
                 theta =inputs[5],
                 b1    =np.random.uniform(0.0,inputs[0],1),
                 b2    =np.random.uniform(0.0,inputs[0],1),
                 b3    =inputs[0],
                 d1    =int_day,
                 d2    =num_days+1,
                 d3    =num_days+1,
                 theta1=np.random.uniform(0.0,inputs[5],1),
                 theta2=inputs[5],
                 theta3=inputs[5])
       k = 0
       for c in cantons_:
          IU0[c] = inputs[6+k]
          E0 [c] = 3*IU0[c]
          k += 1
       S0 = [N - E - IR - IU for N, E, IR, IU in zip(N0, E0, IR0, IU0)]
       y0 = S0 + E0 + IR0 + IU0 + N0
       # Run the ODE solver.
       solver = libepidemics.Solver(data.to_cpp())
       y0 = libepidemics.State(y0)
       return solver.solve(params, y0, t_eval=range(1, num_days + 1))


    if L == 12 + ic_cantons : #inference/model evaluations for case 3 

       params = libepidemics.Parameters(
                 beta  =inputs[0],
                 mu    =inputs[1],
                 alpha =inputs[2],
                 Z     =inputs[3],
                 D     =inputs[4],
                 theta =inputs[5],
                 b1    =inputs[6],
                 b2    =inputs[7],
                 #b3    =np.random.uniform(0.0,0.5,1),
                 b3    =np.random.uniform(0.0,0.03,1),
                 d1    =inputs[8],
                 d2    =inputs[9],
                 d3    =T_S_CASE_3,
                 theta1=inputs[10],
                 theta2=inputs[11],
                 theta3=inputs[5])
       k = 0
       for c in cantons_:
          IU0[c] = inputs[12+k]
          E0 [c] = 3*inputs[12+k]
          k += 1
       S0 = [N - E - IR - IU for N, E, IR, IU in zip(N0, E0, IR0, IU0)]
       y0 = S0 + E0 + IR0 + IU0 + N0
       solver = libepidemics.Solver(data.to_cpp())
       y0 = libepidemics.State(y0)
       return solver.solve(params, y0, t_eval=range(1, num_days + 1))


    if L == 12 + ic_cantons + 1: #inference/model evaluations for case 3b 

       params = libepidemics.Parameters(
                 beta  =inputs[0],
                 mu    =inputs[1],
                 alpha =inputs[2],
                 Z     =inputs[3],
                 D     =inputs[4],
                 theta =inputs[5],
                 b1    =inputs[6],
                 b2    =inputs[7],

                 b3    =inputs[12],
                 d1    =inputs[8],
                 d2    =inputs[9],
                 d3    =102,


                 theta1=inputs[10],
                 theta2=inputs[11],
                 theta3=inputs[5])
       k = 0
       for c in cantons_:
          IU0[c] = inputs[13+k]
          E0 [c] = 3*inputs[13+k]
          k += 1
       S0 = [N - E - IR - IU for N, E, IR, IU in zip(N0, E0, IR0, IU0)]
       y0 = S0 + E0 + IR0 + IU0 + N0
       solver = libepidemics.Solver(data.to_cpp())
       y0 = libepidemics.State(y0)
       return solver.solve(params, y0, t_eval=range(1, num_days + 1))


    if L == 12 + ic_cantons + 2: #inference/model evaluations for case 3b 

       params = libepidemics.Parameters(
                 beta  =inputs[0],
                 mu    =inputs[1],
                 alpha =inputs[2],
                 Z     =inputs[3],
                 D     =inputs[4],
                 theta =inputs[5],
                 b1    =inputs[6],
                 b2    =inputs[7],

                 b3    =inputs[13],
                 d1    =inputs[8],
                 d2    =inputs[9],
                 d3    =inputs[12],


                 theta1=inputs[10],
                 theta2=inputs[11],
                 theta3=inputs[5])
       k = 0
       for c in cantons_:
          IU0[c] = inputs[14+k]
          E0 [c] = 3*inputs[14+k]
          k += 1
       S0 = [N - E - IR - IU for N, E, IR, IU in zip(N0, E0, IR0, IU0)]
       y0 = S0 + E0 + IR0 + IU0 + N0
       solver = libepidemics.Solver(data.to_cpp())
       y0 = libepidemics.State(y0)
       return solver.solve(params, y0, t_eval=range(1, num_days + 1))














