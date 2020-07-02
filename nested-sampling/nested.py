import numpy as np
import pickle
import os
import sys
import argparse
import time

from dynesty import NestedSampler
from multiprocessing import Pool
from data import *
from model import *

refy2_cantons = prepareData(days=21)
refy3_cantons = prepareData(days=102)
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
    parser.add_argument('--nlive',type=int  , default=50 ,help="number of live samples" )
    parser.add_argument('--dlogz',type=float, default=0.1 ,help="dlogz criterion"       )
    parser.add_argument('--cores',type=int  , default=96  ,help="number of cores"       )
    parser.add_argument('--case' ,type=int  , default=2   ,help="2 or 3"                )

    args = parser.parse_args(argv)

    model = model_2
    model_transformation = model_transformation_2
    ndim = 6 + ic_cantons + 1
    if args.case == 3:
       model = model_3
       model_transformation = model_transformation_3
       ndim = 12 + ic_cantons + 1

    t = -time.time()

    fname = basename + str(args.case) + '.pickle'

    pool = MyPool(args.cores)

    sampler = NestedSampler(model,model_transformation,ndim,nlive=args.nlive, bound='multi', pool=pool)

    sampler.run_nested(maxiter=1e8, dlogz=args.dlogz, add_live=True)
    res = sampler.results
    res.summary()
    with open(fname, 'wb') as f:
        pickle.dump(res, f)

    t += time.time()
    print("Total time=",t)
