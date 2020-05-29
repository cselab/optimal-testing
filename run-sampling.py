#!/usr/bin/env python3

import numpy as np
import sys

# Importing the function to run Korali for the sampling
sys.path.append('./sampler')
sys.path.append('./model')
from run_korali_sampler import *
from model import model

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--scenario', '-sc', default='uniform', help='Choose sampling scenario./')
parser.add_argument('--nPoints',  '-np', type=int, default=100, help='Number of points(??).')
parser.add_argument('--sampler',  '-sa', default='TMCMC', help='Choose sampler TMCMC or mTMCMC')
parser.add_argument('--nSamples', '-ns', type=int, default=5000, help='Number of samples for TMCMC.')

priorBounds = [
        [0.2, 3.5, "beta"],
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

if __name__ == "__main__":
    nTime   = 30

    args = parser.parse_args()
    scenario   = args.scenario
    numPoints  = args.nPoints
    numSamples = args.nSamples
    sampler    = args.sampler

    if scenario == "uniform":
            samples = []
            for i in range(6):
                    samples.append(np.random.uniform(priorBounds[i][0], priorBounds[i][1], numPoints))
            np.save("samples.npy", samples)
    elif scenario == "social-distancing":
            samples = runTMCMC(nTime, scenario, sampler, numSamples, priorBounds[:model["nParams"]]+priorBounds[-model["nIC"]:] )
            samples = np.array(samples)
            np.save("samples.npy", samples[np.random.randint(numSamples,size=numPoints)])
    elif scenario == "second-outbreak":
            samples = runTMCMC(nTime, scenario, sampler, numSamples, priorBounds )
            samples = np.array(samples)
            np.save("samples.npy", samples[np.random.randint(numSamples,size=numPoints)])
    else:
        print("Scenario not recognized! Exit...")
