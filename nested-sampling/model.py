from nested import *
from scipy.special import loggamma
sys.path.append(os.path.join(os.path.dirname(__file__), '../covid19/epidemics/cantons/py'))
from run_osp_cases import *

ic_cantons = 12

def model_transformation_2(u):
    x = np.zeros(len(u))
    x[0] = 0.8  + 1.00*u[0]#b0
    x[1] = 0.2  + 0.80*u[1]#mu
    x[2] = 0.02 + 0.98*u[2]#alpha
    x[3] = 1.00 + 5.00*u[3]#Z
    x[4] = 1.00 + 5.00*u[4]#D
    x[5] = 1.00 + 0.40*u[5]#theta
    for i in range (6,6+ic_cantons):
        x[i] = 30*u[i]
    x[6+ic_cantons] = 0.5*u[6+ic_cantons]
    return x
 
def model_2(THETA):
    days = 21 
    par = [THETA[0],THETA[1],THETA[2],THETA[3],THETA[4],THETA[5],
           THETA[0],
           THETA[0],
           THETA[0],
           days,days,days] #no interventions up to day 21
    for i in range (ic_cantons):
        par.append(THETA[6+i])
    results = example_run_seiin(days,par)
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

def model_transformation_3(u):
    x = np.zeros(len(u))
    x[0] = 0.8  + 1.00*u[0]#b0
    x[1] = 0.2  + 0.80*u[1]#mu
    x[2] = 0.02 + 0.98*u[2]#alpha
    x[3] = 1.00 + 5.00*u[3]#Z
    x[4] = 1.00 + 5.00*u[4]#D
    x[5] = 1.00 + 0.40*u[5]#theta
    x[6] = u[6]*x[0]#b1
    x[7] = u[7]*x[0]#b2
    #d1 = 15.0  + 35.0*u[8]
    #d2 = 15.0  + 35.0*u[9]
    d1 = 15.0  + 10.0*u[8]
    d2 = 25.0  + 20.0*u[9]
    x[8] = min(d1,d2) 
    x[9] = max(d1,d2) 
    for i in range (10,10+ic_cantons):
        x[i] = 30*u[i]
    x[10+ic_cantons] = u[10+ic_cantons]*0.5
    return x

def model_3(THETA):
    days = 84 
    par = [THETA[0],THETA[1],THETA[2],THETA[3],THETA[4],THETA[5],
           THETA[6],THETA[7],10000,THETA[8],THETA[9],days]
    for i in range (ic_cantons):
        par.append(THETA[10+i])
    results = example_run_seiin(days,par)
    negativeBinomialConstant = 0
    loglike = 0.0
    for i in range ( 0,len(refy3_cantons),3 ):
            c = refy3_cantons[i  ]
            d = refy3_cantons[i+1]
            cases = results[d].E() 
            m  = THETA[2]/THETA[3]* cases[c] + 1e-16
            r  = THETA[-1]*m
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
