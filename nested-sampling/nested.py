#!/usr/bin/env python3
'''
Nested sampling. Use data to inform the prior distributions used in cases 2 and 3.
'''
import numpy as np
import pickle,os,sys,argparse,time
from dynesty import NestedSampler
from multiprocessing import Pool
from data import *
from scipy.special import loggamma
from seiin import *

ic_cantons = 12
refy2_cantons = prepareData(days=T_DATA_CASE_2)
refy3_cantons = prepareData(days=T_DATA_CASE_3)
refy4_cantons = prepareData(days=T_DATA_CASE_4)

'''
Prior distributions used in the Bayesian inference for cases 2 and 3.
'''
def model_transformation_2(u):
    x = np.zeros(len(u))
    x[0] = 0.8  + 1.00*u[0]#b0
    x[1] = 0.2  + 0.80*u[1]#mu
    x[2] = 0.01 + 0.99*u[2]#alpha
    x[3] = 1.00 + 5.00*u[3]#Z
    x[4] = 1.00 + 5.00*u[4]#D
    x[5] = 0.5  + 1.00*u[5]#theta
    for i in range (6,6+ic_cantons):
        x[i] = 50*u[i]
    x[6+ic_cantons] = 0.5*u[6+ic_cantons]#dispesion
    return x
def model_transformation_3(u):
    x = np.zeros(len(u))
    x[0] = 0.8  + 1.00*u[0]#b0
    x[1] = 0.2  + 0.80*u[1]#mu
    x[2] = 0.01 + 0.99*u[2]#alpha
    x[3] = 1.00 + 5.00*u[3]#Z
    x[4] = 1.00 + 5.00*u[4]#D
    x[5] = 0.5  + 1.0 *u[5]#theta
    x[6] = u[6]*x[0]#b1
    x[7] = u[7]*x[0]#b2
    x[8] = 20.0 + 10.00*u[8]#d1
    x[9] = 30.0 + 10.00*u[9]#d2
    x[10] = u[10]*x[5]#theta 1
    x[11] = u[11]*x[5]#theta 2
    for i in range(12,12+ic_cantons):
        x[i] = 50*u[i]
    x[12+ic_cantons] = u[12+ic_cantons]*0.5#dispersion
    return x

def model_transformation_4(u):
    x = np.zeros(len(u))
    x[0] = 0.8  + 1.00*u[0]#b0
    x[1] = 0.2  + 0.80*u[1]#mu
    x[2] = 0.01 + 0.99*u[2]#alpha
    x[3] = 1.00 + 5.00*u[3]#Z
    x[4] = 1.00 + 5.00*u[4]#D
    x[5] = 0.5  + 1.0 *u[5]#theta
    x[6] = u[6]*x[0]#b1
    x[7] = u[7]*x[0]#b2
    x[8] = 20.0 + 10.00*u[8]#d1
    x[9] = 30.0 + 10.00*u[9]#d2
    x[10] = u[10]*x[5]#theta 1
    x[11] = u[11]*x[5]#theta 2
    
    x[12] = 108 + u[12] #100 + 20*u[12]#d3
    x[13] = 0.03   * u[13]#lambda

    for i in range(14,14+ic_cantons):
        x[i] = 50*u[i]
    x[14+ic_cantons] = u[14+ic_cantons]*0.5#dispersion
    return x

def model_2(THETA):
    days = T_DATA_CASE_2
    results = example_run_seiin(days,THETA[0:len(THETA)-1])
    negativeBinomialConstant = 0
    loglike = 0.0
    for i in range ( 0,len(refy2_cantons),3 ):
            c = refy2_cantons[i  ]
            d = refy2_cantons[i+1]
            cases = results[d].E()
            m  = THETA[2]/THETA[3]* cases[c] + 1e-16
            r = THETA[-1]*m
            if m < 0.0: return -10e32
            yi = refy2_cantons[i+2]
            negativeBinomialConstant -= loggamma(yi+1.)
            p = m/(m+r)
            loglike += loggamma(yi+r)
            loglike -= loggamma( r )
            loglike += r*np.log( 1-p )
            loglike += yi*np.log( p )
    loglike += negativeBinomialConstant
    return loglike
def model_3(THETA):
    days = T_DATA_CASE_3
    results = example_run_seiin(days,THETA[0:len(THETA)-1])
    negativeBinomialConstant = 0
    loglike = 0.0
    for i in range ( 0,len(refy3_cantons),3 ):
            c = refy3_cantons[i  ]
            d = refy3_cantons[i+1]
            cases = results[d].E()
            m  = THETA[2]/THETA[3]* cases[c] + 1e-16
            r = THETA[-1]*m
            if m < 0.0: return -10e32
            yi = refy3_cantons[i+2]
            negativeBinomialConstant -= loggamma(yi+1.)
            p = m/(m+r)
            loglike += loggamma(yi+r)
            loglike -= loggamma( r )
            loglike += r*np.log( 1-p )
            loglike += yi*np.log( p )
    loglike += negativeBinomialConstant
    return loglike

def model_4(THETA):
    days = T_DATA_CASE_4
    results = example_run_seiin(days,THETA[0:len(THETA)-1])
    negativeBinomialConstant = 0
    loglike = 0.0
    for i in range ( 0,len(refy4_cantons),3 ):
            c = refy4_cantons[i  ]
            d = refy4_cantons[i+1]
            cases = results[d].E()
            m  = THETA[2]/THETA[3]* cases[c] + 1e-16
            r = THETA[-1]*m
            if m < 0.0: return -10e32
            yi = refy4_cantons[i+2]
            negativeBinomialConstant -= loggamma(yi+1.)
            p = m/(m+r)
            loglike += loggamma(yi+r)
            loglike -= loggamma( r )
            loglike += r*np.log( 1-p )
            loglike += yi*np.log( p )
    loglike += negativeBinomialConstant
    return loglike


class MyPool(object):
    '''
    Auxiliary function needed to run nested sampling with many cores.
    '''
    def __init__(self, cores):
        self.pool = Pool(processes=cores)
        self.size = cores
    def map(self, function, tasks):
        return self.pool.map(function, tasks)


if __name__=='__main__':

    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--nlive',type=int  , default=50 ,help="number of live samples")
    parser.add_argument('--dlogz',type=float, default=0.1,help="dlogz criterion"       )
    parser.add_argument('--cores',type=int  , default=96 ,help="number of cores"       )
    parser.add_argument('--case' ,type=int  , default=2  ,help="2 or 3"                )
    args = parser.parse_args(argv)

    model = model_2
    model_transformation = model_transformation_2
    ndim = 6 + ic_cantons + 1
    if args.case == 3:
       model = model_3
       model_transformation = model_transformation_3
       ndim = 12 + ic_cantons + 1
    if args.case == 4:
       model = model_4
       model_transformation = model_transformation_4
       ndim = 12 + ic_cantons + 1 + 2

    t = -time.time()
    fname = 'samples_' + str(args.case) + '.pickle'
    pool = MyPool(args.cores)
    sampler = NestedSampler(model,model_transformation,ndim,nlive=args.nlive, bound='multi', pool=pool)
    sampler.run_nested(maxiter=1e8, dlogz=args.dlogz, add_live=True)
    res = sampler.results
    res.summary()
    with open(fname, 'wb') as f:
        pickle.dump(res, f)
    t += time.time()
    print("Total time=",t)
