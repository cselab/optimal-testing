from nested import *
from scipy.special import loggamma
sys.path.append(os.path.join(os.path.dirname(__file__), '../../covid19/epidemics/cantons/py'))
from run_osp_cases import *

ic_cantons = 12

def model_transformation_2(u):
    x = np.zeros(len(u))
    x[0] = 0.8  + 1.0 *u[0]#b0
    x[1] = 0.2  + 0.8 *u[1]#mu
    x[2] = 0.01 + 0.99*u[2]#alpha
    x[3] = 1.00 + 5.00*u[3]#Z
    x[4] = 1.00 + 5.00*u[4]#D
    x[5] = 0.5  + 1.0 *u[5]#theta
    for i in range (6,6+ic_cantons):
        x[i] = 50*u[i]
    x[6+ic_cantons] = 0.5*u[6+ic_cantons]
    return x

def model_transformation_3(u):
    x = np.zeros(len(u))
    x[0] = 0.8  + 1.00*u[0]#b0
    x[1] = 0.2  + 0.8* u[1]#mu
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
    x[12+ic_cantons] = u[12+ic_cantons]*0.5
    return x

def model_2(THETA):
    days = 21
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
    days = 102
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
