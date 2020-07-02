import numpy as np
import pickle,os,sys,argparse,scipy,random
from scipy.special import loggamma
from dynesty import NestedSampler
from multiprocessing import Pool
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../covid19/epidemics/cantons/py'))
from run_osp_cases import *
from scipy.stats import multivariate_normal


from create_data import *

ic_cantons = 12

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]
def prepareData(days):
    data = np.load("files/canton_daily_cases.npy")
    cantons = data.shape[0]
    if days == -1:
    	days = data.shape[1]   
    threshold =.0
    y = []
    for c in range(cantons):
        d_ = np.copy(data[c,:])
        nans, x= nan_helper(d_)
        d_[nans]= np.interp(x(nans), x(~nans), d_[~nans])
        if np.max(d_) < threshold :
            continue
        d1 = np.copy(data[c,:])
        for d in range(days):
            if np.isnan(d1[d]) == False:
                y.append(c)
                y.append(d)
                y.append(d1[d])
    return y


class model_nested:

 def __init__(self,case,data,path_data):

    self.data = data
    self.points = int( len(self.data)/3 )
    self.day_vector    = self.data[0*self.points:1*self.points]
    self.canton_vector = self.data[1*self.points:2*self.points]
    self.case_vector   = self.data[2*self.points:3*self.points]
    if case == 1:
        self.days = 8
        self.ndim = 7 + ic_cantons + 1
        self.p_mle = np.load("files/mle1.npy")
    elif case == 2:
        self.days = 21 + 15
        self.ndim = 14 + ic_cantons 
        self.p_mle = np.load("files/mle2.npy")
    self.case = case
    self.ref_y = prepareData(self.days)
    assert len(self.data) % 3 == 0
    assert case == 1 or case == 2

    self.sigma_mean = np.zeros(self.days)  
    for i in range(self.days):
      temp = np.load(path_data + "tensor_Ntheta={:05d}.npy".format(i))
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


 def transformation1(self,u):   
    x = np.zeros(len(u))
    x[0] = 0.8  + 1.00*u[0]#b0
    x[1] = 0.2  + 0.80*u[1]#mu
    x[2] = 0.01 + 0.99*u[2]#alpha
    x[3] = 1.00 + 5.00*u[3]#Z
    x[4] = 1.00 + 5.00*u[4]#D
    x[5] = 0.50 + 1.00*u[5]#theta
    #x[6] = 0.50       *u[6]#dispersion
    x[6] = 2.0       *u[6]#dispersion
    x[7] = 0.20*u[7]#c
    for i in range (ic_cantons):
        x[8+i] = 50*u[8+i]
    return x

 def transformation2(self,u):   
    assert False
    x = np.zeros(len(u))
    x[0] = 0.80 + 1.00*u[0]#b0
    x[1] = 0.2  + 0.80*u[1]#mu
    x[2] = 0.01 + 0.99*u[2]#alpha
    x[3] = 1.00 + 5.00*u[3]#Z
    x[4] = 1.00 + 5.00*u[4]#D
    x[5] = 0.50 + 1.00*u[5]#theta
    x[6] = x[0]*u[6]#b1
    x[7] = x[0]*u[7]#b2
    x[8] = 21.0 + 3.0*u[8]#d1
    x[9] = 35.0 + 4.0*u[9]#d2
    x[10] = u[10]*x[5]#theta 1
    x[11] = u[11]*x[5]#theta 2
    for i in range(12,12+ic_cantons):
        x[i] = 50*u[i]
    x[12+ic_cantons] =  0.50*u[12+ic_cantons]#dispersion
    x[13+ic_cantons] = 0.05 + 0.20*u[13+ic_cantons]#c
    return x

    return x

 def model(self,THETA):
    dispersion = 0.0
    c_error    = 0.0
    par = [THETA[0],THETA[1],THETA[2],THETA[3],THETA[4],THETA[5]]

    if self.case == 1:
        for i in range (ic_cantons):
            par.append(THETA[8+i])
        dispersion = THETA[6]
        c_error    = THETA[7]
    else:
        for i in range (6,12+ic_cantons):
            par.append(THETA[i])
        dispersion = THETA[24]
        c_error    = THETA[25]

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

class MyPool(object):
    def __init__(self, cores):
        self.pool = Pool(processes=cores)
        self.size = cores
    def map(self, function, tasks):
        return self.pool.map(function, tasks)

def GetNewSamples(cantons,times,path,day_max,case,name,nlive=500,dlogz=0.01,cores=72):
    Iu_all = np.load("data_base.npy")

    Iu = []    
    #store requested measurements
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
    np.save("files/" + name + "_data.npy",aaa)


    M = model_nested(case=case,data=aaa,path_data=path)
    pool = MyPool(cores)
       
    model_transformation = M.transformation1
    if case == 2:
       model_transformation = M.transformation2

    sampler = NestedSampler(M.model, model_transformation, M.ndim, nlive=nlive, bound='multi', pool=pool)

    sampler.run_nested(maxiter=1e7, dlogz=dlogz, add_live=True)
    res = sampler.results
    res.summary()
    fname = name + '.pickle'
    with open("files/"+ fname, 'wb') as f:
        pickle.dump(res, f)







if __name__=='__main__':
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--result',type=str)
    parser.add_argument('--path'  ,type=str)
    parser.add_argument('--case'  ,type=int)
    parser.add_argument('--sensors'  ,type=int)
    parser.add_argument('--optimal'  ,type=int)
    args = parser.parse_args(argv)

    u = np.load("files/"+args.result)
    cantons  = u.shape[1]
    days = 8
    if args.case == 2:
      days = 21 + 10

    SENSORS = args.sensors
    print("Placing",SENSORS,"sensors.")
    #################################################################
    if args.optimal == 1:
     print("Optimal testing")
     c_opt = []
     t_opt = []
     for n in range (SENSORS):
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
     print(t_opt)
     print(c_opt)
     GetNewSamples(cantons=c_opt,times=t_opt,path=args.path,day_max=days,case=args.case,name= "optimal_case"+str(args.case) +"_sensor" + str(SENSORS))
    #################################################################
    else:
     print("Suboptimal testing")
     c_sub = []
     t_sub = []
     ccc1 = [20,3,25,5,7,4,6,9,11,22]
     for i in range (SENSORS):
           c_sub.append(ccc1[i])
           if args.case == 1:
               t_sub.append(3.0)
           elif args.case == 2:
               t_sub.append(23.0)
     print(t_sub)
     print(c_sub)
     GetNewSamples(cantons=c_sub,times=t_sub,path=args.path,day_max=days,case=args.case,name= "uniform_case"+str(args.case) +"_sensor" + str(SENSORS))
    #################################################################
