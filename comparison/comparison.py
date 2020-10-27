import numpy as np
import pickle,os,sys,argparse,scipy,random
from scipy.special import loggamma
from dynesty import NestedSampler
from multiprocessing import Pool
import os.path
from scipy.stats import multivariate_normal
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from seiin import *
import swiss_cantons

ic_cantons = 12

def distance(t1,t2,tau):
    dt = np.abs(t1-t2) / tau
    return np.exp(-dt)

def covariance_matrix(space,time,sigma_mean):
    tau_real = 2.0
    days = len(space)
    cov = np.zeros((days,days))
    for i in range(days):
        for j in range(days):
            t1 = time [i]
            t2 = time [j] 
            s1 = space[i] 
            s2 = space[j] 
            if s1 == s2:
               coef = distance(t1,t2,tau_real)  
               #Small hack. When coef --> 1, two measurements are correlated and should not be both made
               #If coef is not explicitly set to 1.0, we get covariance matrices that are ill-conditioned (det(cov)--> 0)
               #and the results are weird. This hack guarantees numerical stability by explicitly making the covariance
               #exactly singular.
               if coef > 0.99: 
                  coef = 1.0
               cov[i,j] = (sigma_mean[i]*sigma_mean[j])*coef
    return cov

def GenerateData(day_max):
    np.random.seed(12345)
    sigma_mean = np.zeros(day_max)  
    for i in range(day_max):
      temp = np.load("../case1/day={:05d}.npy".format(i))
      sigma_mean[i] = np.mean(temp.flatten())

    #model run with maximum a posteriori estimate
    p_ = []    
    p = np.load("../case2/map.npy")
    p_ = [p[0],p[1],p[2],p[3],p[4],p[5]]
    for i in range(ic_cantons):
       p_.append(p[6+i])
    r = example_run_seiin(day_max,p_,1000)

    Iu_all = np.zeros((day_max,26))

    c_real = 0.1
    time   = np.arange(day_max)
    space  = np.zeros(day_max)
    aux    = covariance_matrix(space,time,sigma_mean)
    COVARIANCE = (c_real*c_real)*aux
    for c1 in range(26):
        mean = np.zeros(day_max)
        for d1 in range(day_max):
            mean[d1] = r[d1].Iu()[c1]
        rv = scipy.stats.multivariate_normal(mean=mean, cov=COVARIANCE, allow_singular=True)
        Iu_all[:,c1] = rv.rvs(size=1)
    Iu_all[np.where(Iu_all<0)] = 0.0
    np.save("data_base.npy",Iu_all)


class model_nested:
 def __init__(self,data):
    self.data = data
    self.points = int( len(self.data)/3 )
    self.day_vector    = self.data[0*self.points:1*self.points]
    self.canton_vector = self.data[1*self.points:2*self.points]
    self.case_vector   = self.data[2*self.points:3*self.points]

    self.days = 8
    self.ndim = 7 + ic_cantons + 1
    self.p_mle = np.load("../case2/map.npy")
    self.ref_y = swiss_cantons.prepareData(self.days)
    assert len(self.data) % 3 == 0

    self.sigma_mean = np.zeros(self.days)  
    for i in range(self.days):
      temp = np.load("../case1/day={:05d}.npy".format(i))
      self.sigma_mean[i] = np.mean(temp.flatten())

    sigma_mean = np.zeros(self.points)
    for i in range(self.points):
       sigma_mean[i] = self.sigma_mean[int(self.day_vector[i])]
    aux = np.zeros((self.points,self.points))
    for i in range(self.points):
          for j in range(self.points):
              t1 = self.day_vector[i]
              t2 = self.day_vector[j]
              s1 = self.canton_vector[i]
              s2 = self.canton_vector[j]
              if s1 == s2:
                 coef = distance(t1,t2,2.0)  
                 #Small hack. When coef --> 1, two measurements are correlated and should not be both made
                 #If coef is not explicitly set to 1.0, we get covariance matrices that are ill-conditioned (det(cov)--> 0)
                 #and the results are weird. This hack guarantees numerical stability by explicitly making the covariance
                 #exactly singular.
                 if coef > 0.99: 
                    coef = 1.0
                 aux[i,j] = (sigma_mean[i]*sigma_mean[j])*coef
              else:
                 aux[i,j] = 0.0 
    self.COVARIANCE = aux

 #prior distributions for model parameters
 def transformation(self,u):   
    x = np.zeros(len(u))
    x[0] = 0.8  + 1.00*u[0]#b0
    x[1] = 0.2  + 0.80*u[1]#mu
    x[2] = 0.01 + 0.99*u[2]#alpha
    x[3] = 1.00 + 5.00*u[3]#Z
    x[4] = 1.00 + 5.00*u[4]#D
    x[5] = 0.50 + 1.00*u[5]#theta
    x[6] = 2.0       *u[6]#dispersion
    x[7] = 0.20*u[7]#c
    for i in range (ic_cantons):
        x[8+i] = 50*u[8+i]
    return x

 def model(self,THETA):
    dispersion = 0.0
    c_error    = 0.0
    par = [THETA[0],THETA[1],THETA[2],THETA[3],THETA[4],THETA[5]]

    for i in range (ic_cantons):
        par.append(THETA[8+i])
    dispersion = THETA[6]
    c_error    = THETA[7]

    results = example_run_seiin(self.days,par)
    loglike = 0.0
    mean = np.zeros(self.points)
    Y    = np.zeros(self.points)
    for i in range (self.points):
        c    = int(self.canton_vector[i])
        d    = int(self.day_vector[i])
        Y[i] = self.case_vector[i] 
        mean[i] = results[d].Iu()[c]
    covariance = (c_error*c_error)*self.COVARIANCE 
    loglike += multivariate_normal.logpdf(Y, mean=mean, cov=covariance)

    coef = THETA[2]/THETA[3]
    negativeBinomialConstant = 0
    for i in range (0,len(self.ref_y),3):
        c     = self.ref_y[i  ]
        d     = self.ref_y[i+1]
        yi    = self.ref_y[i+2]
        cases = results[d].E()             
        m     = coef * cases[c] + 1e-16
        r     = dispersion*m
        p     = m/(m+r)
        assert m >= 0.0
        negativeBinomialConstant -= loggamma(yi+1.)
        loglike += loggamma(yi+r) - loggamma( r ) + r*np.log( 1-p ) + yi*np.log( p )
    loglike += negativeBinomialConstant

    return loglike

#auxiliary function for sampling with Dynesty
class MyPool(object):
    def __init__(self, cores):
        self.pool = Pool(processes=cores)
        self.size = cores
    def map(self, function, tasks):
        return self.pool.map(function, tasks)

def GetNewSamples(cantons,times,day_max,name,nlive=500,dlogz=0.01,cores=12):
    Iu_all = np.load("data_base.npy")

    Iu = []    
    #store requested surveys
    for i in range (len(cantons)):
        canton = int(cantons[i])
        day = int(times[i])
        Iu.append(Iu_all[day,canton])
    aaa = []
    for i in range(len(cantons)):
        aaa.append(times[i])
    for i in range(len(cantons)):
        aaa.append(cantons[i])
    for i in range(len(cantons)):
        aaa.append(Iu[i])
    aaa = np.asarray(aaa)
    np.save(name + "_data.npy",aaa)

    #Bayesian inference using the results of the (artificial) surveys
    M = model_nested(data=aaa)
    pool = MyPool(cores)    
    sampler = NestedSampler(M.model, M.transformation, M.ndim, nlive=nlive, bound='multi', pool=pool)
    sampler.run_nested(maxiter=1e7, dlogz=dlogz, add_live=True)
    res = sampler.results
    res.summary()
    fname = name + '.pickle'
    with open(fname, 'wb') as f:
        pickle.dump(res, f)

if __name__=='__main__':
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--surveys',type=int)
    args = parser.parse_args(argv)
    surveys = args.surveys

    #Generate artificial data for the first 20 days of the epidemic 
    GenerateData(day_max=20)

    #Load the utility function that was computed for case 1, for the first 8 days of the epidemic
    u = np.load("../case1/result.npy")
    days = 8

    #For the optimal strategy, find the maximum utility 
    c_opt = []
    t_opt = []
    for n in range (surveys):
        utility = u[n,:,:]
        dopt = -1
        copt = -1
        umax = 0.0
        for d in range(days):
            for c in range(26):
                if utility[c,d]>umax:
                    dopt = d
                    copt = c
                    umax = utility[c,d]
        c_opt.append(copt)
        t_opt.append(dopt)
    #Do the surveys according to the max utility
    GetNewSamples(cantons=c_opt,times=t_opt,day_max=days,name= "optimal_surveys" + str(surveys))

    #For the non-specific strategy, test the cantons of Ticino (20), Bern (3), Zurich (25) and so on
    #All surveys take place on the 3rd day
    c_sub = []
    t_sub = []
    ccc1 = [20,3,25,5,7,4,6,9,11,22]
    for i in range (surveys):
      c_sub.append(ccc1[i])
      t_sub.append(3.0)

    #Do the surveys according to non-specific strategy
    GetNewSamples(cantons=c_sub,times=t_sub,day_max=days,name= "nonspecific_surveys" + str(surveys))