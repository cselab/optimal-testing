import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import time

from scipy import signal
from scipy.special import loggamma
from dynesty import NestedSampler
from pandas.plotting import scatter_matrix
from dynesty import plotting as dyplot

from data import *
from model import *
from multiprocessing import Pool

cantons = True
refy2_cantons = prepareData(days=21)
refy3_cantons = prepareData()
basename = "cantons___"
ndim = 0

class MyPool(object):
    def __init__(self, cores):
        self.pool = Pool(processes=cores)
        self.size = cores
    def map(self, function, tasks):
        return self.pool.map(function, tasks)

if __name__=='__main__':

    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('--nlive',type=int  , default=1000 ,help="number of live samples")
    parser.add_argument('--dlogz',type=float, default=0.1  ,help="dlogz criterion"       )
    parser.add_argument('--cores',type=int  , default=12   ,help="number of cores"       )
    parser.add_argument('--case' ,type=int  , default=2    ,help="2 or 3"                )

    args = parser.parse_args(argv)

    model = model_2
    model_transformation = model_transformation_2
    ndim = 6 + ic_cantons + 1
    if args.case == 3:
       model = model_3
       model_transformation = model_transformation_3
       ndim = 10 + ic_cantons + 1
       
    t = -time.time()

    fname = basename + '.pickle'
 
    pool = MyPool(args.cores)
    
    sampler = NestedSampler(model,model_transformation,ndim,nlive=args.nlive, bound='multi', pool=pool)

    for i in range (1,100):
        print ("===============================")
        print ("Running iteration number:",i,flush=True)
        print ("===============================")
        sampler.run_nested(maxiter=1000, dlogz=args.dlogz, add_live=True)

        res = sampler.results
        res.summary()

        with open(fname, 'wb') as f:
            pickle.dump(res, f)

    t += time.time()
    print("Total time=",t)