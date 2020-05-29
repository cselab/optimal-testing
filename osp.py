import argparse
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
import itertools

class OSP:

  #############################################################################################
  def __init__(self, path, nSensors = 1, nMeasure = 1, Ntheta = 100, Ny = 100, korali = False,start_day = -1):
  #############################################################################################
    self.path     = path      # path to output.npy
    self.nSensors = nSensors  # how many sensors to place
    self.nMeasure = nMeasure  # how many quantities each sensor measures
    self.Ny       = Ny        # how many samples to draw when evaluating utility
    self.Ntheta   = Ntheta    # how many model simulations were done

    self.korali   = korali    # whether Korali will be use for the sensor placement or not

    #output.npy should contain a 3D numpy array [Simulations][Time][Space]
    self.data        = np.load(path+"/output_Ntheta={:05d}.npy".format(Ntheta))
    #self.parameters  = np.load(path+"/params_Ntheta={:05d}.npy".format(Ntheta))

    #arbitrary choices for time correlation length and error model
    self.l     = 3
    self.sigma = 0.2 
    self.sigma_mean = self.sigma * np.mean ( np.mean(self.data,axis=0) , axis = 1)

    assert nMeasure == 1 #probably nMeasure > 1 does not work yet 

    self.current_day = -1
    self.current_canton = -1
    self.start_day = start_day

  
  #############################################################################################
  def EvaluateUtility(self,space_time):
  #############################################################################################
    #time is an array containing the time instance of each one of the nSensors measurements
    #space contains the place where each measurement happens
    space = []
    time  = []
    if self.korali:
      st = space_time["Parameters"]
      assert( len(st)%2 == 0 )
      n = int ( len(st)/ 2 )
      for i in range(n*2):
        if i%2 == 0:
          space.append(int(st[-(i+1)]))
        else:
          time.append(int(st[-(i+1)]))
    else:
      n = int ( len(space_time)/2 )
      for i in range(n):
        space.append(space_time[i])
        time.append(space_time[i+n])

    #for i in range(n):
    #    if time[i] < 63:
    #       space_time["F(x)"] = 0.0
    #       return

    for i in range(n-1):
        if time[n-1] == time[i] and space[n-1] == space[i]:
           st = []
           for j in range(n-1):
               st.append(space[j])
           for j in range(n-1):
               st.append(time[j])
           

           self.korali = False
           retval = self.EvaluateUtility(st)
           self.korali = True
           if self.korali:
              space_time["F(x)"] = retval
              return 
           else:
              return retval

    M = self.nMeasure
    N = len(time)
    F = np.zeros((self.Ntheta,  M*N ))
    for s in range(0,N):
      F[:,s*M:(s+1)*M] = self.data[:,time[s], self.nMeasure*space[s] : self.nMeasure*(space[s]+1) ] 

    #Estimate covariance matrix as a function of the sensor locations (time and space)
    T1,T2 = np.meshgrid(time ,time )
    X1,X2 = np.meshgrid(space,space)
    block = np.exp( -self.distance(T1,X1,T2,X2) ) 
    cov   = np.kron(np.eye(self.nMeasure), block)



    sigma_mean = np.zeros(N)
    for i in range(N):
      sigma_mean[i] = self.sigma_mean[time[i]]
    
    #compute utility
    retval = 0.0
    for theta in range(0,self.Ntheta):

      mean = F[theta][:]
      sig  = sigma_mean * np.eye(N)
      rv   = sp.multivariate_normal(np.zeros(n), sig*cov*sig,allow_singular=True)
      y    = np.random.multivariate_normal(mean, sig*cov*sig, self.Ny)  
      s1   = np.mean(rv.logpdf(y-mean))    
      
      #this is a faster way to avoid a second for loop over Ntheta
      m1,m2 = np.meshgrid(y[:,0],F[:,0] )
      m1 = m1.reshape((m1.shape[0]*m1.shape[1],1))
      m2 = m2.reshape((m2.shape[0]*m2.shape[1],1))
      for ns in range(1,n):
        m1_tmp,m2_tmp = np.meshgrid(y[:,ns],F[:,ns] )
        m1_tmp = m1_tmp.reshape(m1_tmp.shape[0]*m1_tmp.shape[1],1)
        m2_tmp = m2_tmp.reshape(m2_tmp.shape[0]*m2_tmp.shape[1],1)
        m1=np.concatenate( (m1,m1_tmp), axis= 1 )
        m2=np.concatenate( (m2,m2_tmp), axis= 1 )
      Pdf_inner = np.empty((self.Ntheta*self.Ny))
      Pdf_inner = rv.pdf(m1-m2)
      Pdf_inner = Pdf_inner.reshape((self.Ntheta,self.Ny))

      s2 = np.mean ( np.log( np.mean( Pdf_inner,axis=0) ) )
      retval += (s1-s2)

    retval /= self.Ntheta
    if self.korali:
      space_time["F(x)"] = retval
    else:
      return retval

  #############################################################################################
  def distance(self,time1,space1,time2,space2):
  #############################################################################################
    s = space1 - space2
    s[ s != 0  ] = np.iinfo(np.int32).max #assume different locations in space are uncorrelated
    retval = np.abs(time1-time2)/self.l + s 
    return retval

  #############################################################################################
  def index(self,x,d):
  #############################################################################################
    index = 0
    for i in range(0,self.nSensors):
      index += x[i] * d**i 
    return index

  #############################################################################################
  def Sequential_Placement(self):
  #############################################################################################
    np.random.seed(12345)
    t_locations = self.data.shape[1]
    x_locations = int(self.data.shape[2] / self.nMeasure)
    t = np.arange(0,t_locations)
    x = np.arange(0,x_locations)

    t_sensors = []
    x_sensors = []
    v_sensors = []
    self.v_all = []

    counter = 0

    print ("Total evaluations required= ",t_locations*x_locations*self.nSensors )

    for n in range(self.nSensors):

      print ("Placing sensor",n+1,"of",self.nSensors)

      vmax = -1e50
      xmax = -1
      tmax = -1

      t_sensors.append(tmax)
      x_sensors.append(xmax)
      v_sensors.append(vmax)

      v_loc1 = []

      for s_x in x:
        
        v_loc2 = []
        for s_t in t:

          counter += 1

          t_sensors[n] = s_t
          x_sensors[n] = s_x

          v = self.EvaluateUtility(x_sensors + t_sensors)

          print (counter,"x=",s_x,"t=",s_t,"utility=",v)
          
          if v > vmax:
            t_ok = True
            x_ok = True
            for i in range (n-1):
              t_ok = (t_sensors[i] != s_t)
              x_ok = (x_sensors[i] != s_x)
            if x_ok and t_ok:
              vmax = v
              xmax = s_x
              tmax = s_t
      
          v_loc2.append(v)
        v_loc1.append(v_loc2)
      self.v_all.append(v_loc1)

      t_sensors[n] = tmax
      x_sensors[n] = xmax
      v_sensors[n] = vmax

      print ("Placed sensor",n,"at x=",xmax,"t=",tmax,'v=',self.EvaluateUtility(x_sensors + t_sensors))

    max_v = self.EvaluateUtility(x_sensors + t_sensors)

    print ("Maximum utility:", max_v)
    print ("Optimal sensor locations")
    for n in range (self.nSensors):
      print ("Sensor",n,"in location ",x_sensors[n],"at time",t_sensors[n])

    np.save("result_time.npy", t_sensors)
    np.save("result_space.npy", x_sensors)
    np.save("result.npy", self.v_all)
    
    return t_sensors,x_sensors



  #############################################################################################
  def EvaluateUtility1(self,argument):
  #############################################################################################
    space = []
    time  = []
    if self.korali:
      st = argument["Parameters"]
      n  = len(st) 
      for i in range(n):
          space.append(int(st[-(i+1)])) #space.append(int(st[i]))
          time.append(self.current_day)
    else:
      n =  len(argument) 
      for i in range(n):
        space.append(argument[i])
        time.append(self.current_day)

    n = len(space)

    for i in range(n-1):
        if space[n-1] == space[i]:
           argument["F(x)"] = -666
           #print("!!",i,space,argument["F(x)"])
           return 

           st = []
           for j in range(n-1):
               st.append(space[j])

           self.korali = False
           retval = self.EvaluateUtility1(st)
           self.korali = True
           if self.korali:
              argument["F(x)"] = retval
              return 
           else:
              return retval

    M = self.nMeasure
    N = len(time)
    F = np.zeros((self.Ntheta,  M*N ))
    for s in range(0,N):
      F[:,s*M:(s+1)*M] = self.data[:,time[s], self.nMeasure*space[s] : self.nMeasure*(space[s]+1) ] 

    #Estimate covariance matrix as a function of the sensor locations (time and space)
    T1,T2 = np.meshgrid(time ,time )
    X1,X2 = np.meshgrid(space,space)
    block = np.exp( -self.distance(T1,X1,T2,X2) ) 
    cov   = np.kron(np.eye(self.nMeasure), block)

    sigma_mean = np.zeros(N)
    for i in range(N):
      sigma_mean[i] = self.sigma_mean[time[i]]
    
    #compute utility
    retval = 0.0
    for theta in range(0,self.Ntheta):

      mean = F[theta][:]
      sig  = sigma_mean * np.eye(N)
      rv   = sp.multivariate_normal(np.zeros(n), sig*cov*sig,allow_singular=True)
      y    = np.random.multivariate_normal(mean, sig*cov*sig, self.Ny)  
      s1   = np.mean(rv.logpdf(y-mean))    
      
      #this is a faster way to avoid a second for loop over Ntheta
      m1,m2 = np.meshgrid(y[:,0],F[:,0] )
      m1 = m1.reshape((m1.shape[0]*m1.shape[1],1))
      m2 = m2.reshape((m2.shape[0]*m2.shape[1],1))
      for ns in range(1,n):
        m1_tmp,m2_tmp = np.meshgrid(y[:,ns],F[:,ns] )
        m1_tmp = m1_tmp.reshape(m1_tmp.shape[0]*m1_tmp.shape[1],1)
        m2_tmp = m2_tmp.reshape(m2_tmp.shape[0]*m2_tmp.shape[1],1)
        m1=np.concatenate( (m1,m1_tmp), axis= 1 )
        m2=np.concatenate( (m2,m2_tmp), axis= 1 )
      Pdf_inner = np.empty((self.Ntheta*self.Ny))
      Pdf_inner = rv.pdf(m1-m2)
      Pdf_inner = Pdf_inner.reshape((self.Ntheta,self.Ny))

      s2 = np.mean ( np.log( np.mean( Pdf_inner,axis=0) ) )
      retval += (s1-s2)

    retval /= self.Ntheta
    if self.korali:
      #print(space,retval)
      argument["F(x)"] = retval
    else:
      return retval


  #############################################################################################
  def EvaluateUtility2(self,argument):
  #############################################################################################
    space = []
    time  = []
    if self.korali:
      st = argument["Parameters"]
      n  = len(st) 
      for i in range(n):
          time.append(int(st[-(i+1)]))
          space.append(self.current_canton)
    else:
      n =  len(argument) 
      for i in range(n):
        time.append(argument[i])
        space.append(self.current_canton)

    n = len(time)
    for i in range(n):
        if time[i] < self.start_day:
          argument["F(x)"] = 0.0
          return

    for i in range(n-1):
        if time[n-1] == time[i]:

           st = []
           for j in range(n-1):
               st.append(time[j])

           self.korali = False
           retval = self.EvaluateUtility2(st)
           self.korali = True
           if self.korali:
              argument["F(x)"] = retval
              return 
           else:
              return retval

    M = self.nMeasure
    N = len(time)
    F = np.zeros((self.Ntheta,  M*N ))
    for s in range(0,N):
      F[:,s*M:(s+1)*M] = self.data[:,time[s], self.nMeasure*space[s] : self.nMeasure*(space[s]+1) ] 

    #Estimate covariance matrix as a function of the sensor locations (time and space)
    T1,T2 = np.meshgrid(time ,time )
    X1,X2 = np.meshgrid(space,space)
    block = np.exp( -self.distance(T1,X1,T2,X2) ) 
    cov   = np.kron(np.eye(self.nMeasure), block)

    sigma_mean = np.zeros(N)
    for i in range(N):
      sigma_mean[i] = self.sigma_mean[time[i]]
    
    #compute utility
    retval = 0.0
    for theta in range(0,self.Ntheta):

      mean = F[theta][:]
      sig  = sigma_mean * np.eye(N)
      rv   = sp.multivariate_normal(np.zeros(n), sig*cov*sig,allow_singular=True)
      y    = np.random.multivariate_normal(mean, sig*cov*sig, self.Ny)  
      s1   = np.mean(rv.logpdf(y-mean))    
      
      #this is a faster way to avoid a second for loop over Ntheta
      m1,m2 = np.meshgrid(y[:,0],F[:,0] )
      m1 = m1.reshape((m1.shape[0]*m1.shape[1],1))
      m2 = m2.reshape((m2.shape[0]*m2.shape[1],1))
      for ns in range(1,n):
        m1_tmp,m2_tmp = np.meshgrid(y[:,ns],F[:,ns] )
        m1_tmp = m1_tmp.reshape(m1_tmp.shape[0]*m1_tmp.shape[1],1)
        m2_tmp = m2_tmp.reshape(m2_tmp.shape[0]*m2_tmp.shape[1],1)
        m1=np.concatenate( (m1,m1_tmp), axis= 1 )
        m2=np.concatenate( (m2,m2_tmp), axis= 1 )
      Pdf_inner = np.empty((self.Ntheta*self.Ny))
      Pdf_inner = rv.pdf(m1-m2)
      Pdf_inner = Pdf_inner.reshape((self.Ntheta,self.Ny))

      s2 = np.mean ( np.log( np.mean( Pdf_inner,axis=0) ) )
      retval += (s1-s2)

    retval /= self.Ntheta
    if self.korali:
      #print(space,retval)
      argument["F(x)"] = retval
    else:
      return retval
