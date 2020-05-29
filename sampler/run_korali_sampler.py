#!/usr/bin/env python3

# Import Korali
import korali

# Importing the used functions for the computational model
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../model'))
from model import *

from epidemics.cantons.py.model import get_canton_model_data

# In this function we invoque Korali to sample the posterior distribution
def runTMCMC(ntime, scenario, sampler, numSamples, priorBounds ):
	
    # Load data
    data = get_canton_model_data()
        
    # Creating new experiment
    e = korali.Experiment()
        
    # Setting up the reference likelihood for the Bayesian Problem
    e["Problem"]["Type"] = "Bayesian/Reference"
    e["Problem"]["Likelihood Model"] = "Normal"
    e["Problem"]["Reference Data"] = getReferenceData( ntime )
    e["Problem"]["Computational Model"] = lambda sample: runCantonsSEIIN(sample, data, ntime, sampler, scenario, getReferencePoints( ntime ))

    # Configuring TMCMC parameters
    e["Solver"]["Type"] = "TMCMC"
    e["Solver"]["Version"] = sampler
    e["Solver"]["Population Size"] = numSamples
    e["Console Output"]["Verbosity"] = 'Detailed'

    # Configuring the problem's random distributions and variables
    i = 0
    for bounds in priorBounds:
            e["Distributions"][i]["Name"] = "Uniform {}".format(i)
            e["Distributions"][i]["Type"] = "Univariate/Uniform"
            e["Distributions"][i]["Minimum"] = bounds[0]
            e["Distributions"][i]["Maximum"] = bounds[1]

            e["Variables"][i]["Name"] = bounds[2]
            e["Variables"][i]["Prior Distribution"] = "Uniform {}".format(i)
            i += 1

    e["Distributions"][i]["Name"] = "Uniform Sigma"
    e["Distributions"][i]["Type"] = "Univariate/Uniform"
    e["Distributions"][i]["Minimum"] = +1e-4
    e["Distributions"][i]["Maximum"] = +100
    
    e["Variables"][i]["Name"] = "Sigma"
    e["Variables"][i]["Prior Distribution"] = "Uniform Sigma"

    # Starting Korali's Engine and running experiment
    print("Starting Korali")
    k = korali.Engine()
    k.run(e)
    print("Finished Korali")

    # Return sample database
    return e["Solver"]["Sample Database"]
