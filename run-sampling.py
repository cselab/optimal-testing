#!/usr/bin/env python3

import numpy as np
import sys

# Importing the function to run Korali for the sampling
sys.path.append('./sampler')
from run_korali_sampler import *

nTime = 30
nSamples = 100
useMTMCMC = False # else TMCMC
nParams = 6
nIC = 12

priorBounds = [ [0.2, 3.5, "beta"],
				[0,   1.4, "mu"],
				[0,   1  , "alpha"],
				[1,   10 , "Z"],
				[1,   10 , "D"],
				[0.8, 4.0, "theta"],
				[0.2, 3.5, "b0"],
				[0.2, 3.5, "b1"],
				[10,   80, "d1"],
				[10,   80, "d2"],
				[0,  1000, "IC_AG"],
				[0,  1000, "IC_BE"],
				[0,  1000, "IC_BL"],
				[0,  1000, "IC_BS"],
				[0,  1000, "IC_FR"],
				[0,  1000, "IC_GE"],
				[0,  1000, "IC_GR"],
				[0,  1000, "IC_SG"],
				[0,  1000, "IC_TI"],
				[0,  1000, "IC_VD"],
				[0,  1000, "IC_VS"],
				[0,  1000, "IC_ZH"]
			  ]

if sys.argv[1] == "uniform":
	samples = []
	for i in range(6):
		samples.append(np.random.uniform(priorBounds[i][0], priorBounds[i][1],nSamples))
	np.save("samples.npy", samples)
elif sys.argv[1] == "social-distancing":
	samples = runTMCMC(nTime, sys.argv[1], useMTMCMC, priorBounds[:nParams]+priorBounds[-nIC:] )
	samples = np.array(samples)
	np.save("samples.npy", samples[np.random.randint(5000,size=nSamples)])
elif sys.argv[1] == "second-outbreak":
	samples = runTMCMC(nTime, sys.argv[1], useMTMCMC, priorBounds )
	samples = np.array(samples)
	np.save("samples.npy", samples[np.random.randint(5000,size=nSamples)])

