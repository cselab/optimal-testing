import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import argparse
import datetime
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
from matplotlib.dates import DateFormatter
import matplotlib
from datetime import timedelta
sys.path.append(os.path.join(os.path.dirname(__file__), '../../covid19/epidemics/cantons/py'))
from run_osp_cases import *
from nested_plot import *


import matplotlib.colors as mcolors
COLORS = mcolors.CSS4_COLORS


ic_cantons=12
cantons = 26
p_conf = 0.99
name = ['AG','AI','AR','BE','BL','BS','FR','GE','GL','GR','JU','LU','NE',\
        'NW','OW','SG','SH','SO','SZ','TG','TI','UR','VD','VS','ZG','ZH']
    
#p_mle_2 = np.load("files/mle2.npy")
#d1 = p_mle_2[8]
#d2 = p_mle_2[9]

def distance(t1,t2,tau):
    dt = np.abs(t1-t2) / tau
    return np.exp(-dt)
####################################################################################################
def getPosteriorFromResult(result):
####################################################################################################
    from dynesty import utils as dyfunc
    weights = np.exp(result.logwt - result.logz[-1]) #normalized weights
    samples = dyfunc.resample_equal(result.samples, weights) #Compute 10%-90% quantiles.
    return samples
####################################################################################################
def model(days,THETA,case):
####################################################################################################
  ic_cantons = 12
  if case == 1:
    par = [THETA[0],THETA[1],THETA[2],THETA[3],THETA[4],THETA[5]] 
    for i in range (ic_cantons):
        par.append(THETA[8+i])
    return example_run_seiin(days,par,1000)
  elif case == 2:
    par = []
    for i in range (12+ic_cantons):
        par.append(THETA[i])
    return example_run_seiin(days,par)
####################################################################################################
def confidence_intervals_CH(results,fname,sensors,m=1,case=1):
####################################################################################################	
    p_mle = np.load("files/mle"+str(case)+".npy")
    days = 21 - 7
    if case == 2:
       days = 21 + 15 + 5
    base  = datetime.datetime(2020, 2, 25) #February 25th, 2020
    dates = np.array([base + datetime.timedelta(hours=(24 * i)) for i in range(days)])
    #locator   = mdates.WeekdayLocator(interval=1)
    #locator2  = mdates.WeekdayLocator(interval=2)
    locator   = mdates.DayLocator(interval=1)
    locator2  = mdates.DayLocator(interval=4)
    date_form = DateFormatter("%b %d")

    #1.Find reference solution
    exact = np.zeros ( (cantons, days) )
    if case == 1:
      p = [p_mle[0],p_mle[1],p_mle[2],p_mle[3],p_mle[4],p_mle[5]]
      for i in range (ic_cantons):
        p.append(p_mle[6+i])
      print("MAP=",p)
      r = example_run_seiin(days,p,1000)
      for i in range(days):
          exact[:,i] = np.asarray(r[i].Iu())
    elif case == 2:
      r = example_run_seiin(days,p_mle[0:len(p_mle)-1])
      for i in range(days):
          exact[:,i] = np.asarray(r[i].Iu())

    #2.Plot country and cantons results  
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(locator2)    
    ax.xaxis.set_minor_locator(locator)    
    ax.xaxis.set_major_formatter(date_form)
    ax.grid()
    ax.plot(dates ,np.sum(exact,axis=0),".",color="black",label='reference')

    #3.Run model for all samples of theta
    #colors = ['red','blue']
    colors = [COLORS['mediumorchid'],COLORS['silver']] 
    Names = ['optimal testing','sub-optimal testing']
    prediction_matrix = []  
    jj = -1
    for result in results:
        jj += 1
        samples = getPosteriorFromResult(result)
        prediction_matrix.append(np.zeros ( (samples.shape[0]//m, days, cantons ) ))
        for i in range(  samples.shape[0]//m):
           print (i,"/",samples.shape[0]//m,flush=True)
           simulation = model(days,samples[i,:],case)
           prediction = []
           for d in range ( days ):
               cases = simulation[d].Iu()
               prediction_matrix [jj][i,d] = np.asarray(cases)

    fig2, axs   = plt.subplots(6,5)
    jj = -1
    for result in results:
      jj += 1
      Name = Names[jj]
      color = colors[jj]

      samples = getPosteriorFromResult(result)
      
      #Plot results for all of Switzerland
      qlo  = np.quantile ( a= np.sum(prediction_matrix[jj],axis=2) , q = 0.5 - p_conf/2 , axis = 0)
      qhi  = np.quantile ( a= np.sum(prediction_matrix[jj],axis=2) , q = 0.5 + p_conf/2 , axis = 0)
      ax.fill_between(dates, qlo, qhi,label=Name + ' ' + str(100*p_conf)+"% confidence interval",alpha=0.2,color=color)

      #Plot for individual cantons
      qlo = np.quantile ( a= prediction_matrix[jj], q = 0.5 - p_conf/2 , axis = 0)
      qhi = np.quantile ( a= prediction_matrix[jj], q = 0.5 + p_conf/2 , axis = 0)
      q50 = np.quantile ( a= prediction_matrix[jj], q = 0.5            , axis = 0)
      for i0 in range (6):
        for i1 in range (5):
          index = i0 * 5 + i1
          if index <= 25:
            axs[i0,i1].fill_between(dates, qlo[:,index], qhi[:,index],label=Name + ' ' + str(100*p_conf)+"% confidence interval",alpha=0.2,color=color)
            axs[i0,i1].text(.5,1.05,name[index],horizontalalignment='center',transform=axs[i0,i1].transAxes)
            axs[i0,i1].xaxis.set_major_locator(locator2)  
            axs[i0,i1].xaxis.set_minor_locator(locator)    
            axs[i0,i1].xaxis.set_major_formatter(date_form)
            axs[i0,i1].set(xlabel='', ylabel=r'$I^u$')
            if jj == 1:
              axs[i0,i1].plot(dates ,exact[index,:],".",color="black",label='reference') 
            for label in axs[i0,i1].get_xticklabels():
                label.set_rotation(40)
                label.set_horizontalalignment('right')

    for i0 in range (6):
      for i1 in range (5):
        index = i0 * 5 + i1
        if index > 25:
          fig2.delaxes(axs[i0][i1])
        else:
          axs[i0,i1].grid()   

    #handles, labels = axs[4,1].get_legend_handles_labels()
    #fig2.legend(handles, labels, loc='lower center',ncol=2,bbox_to_anchor=(0.6, 0.1))
    fig2.set_size_inches(14.5, 10.5)
    plt.tight_layout()

    nnn = "_case" + str(case) + "_sensors" + str(sensors) 

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #      fancybox=True, shadow=True, ncol=2,prop={'size': 8})
    ax.tick_params(labelsize=8)


    fig.savefig ("Iu_country" +nnn+ ".pdf",dpi=1000)
    fig2.savefig("model.pdf",dpi=1000)

    print("Plotting error + model confidence intervals...")
    days = 21 - 7
    if case == 2:
       days = 38

    print ("Reading data for error model...")
    path = "../nested-sampling/case" + str(case) +"/"
    sigma_mean  = np.zeros(days)  
    for i in range(days):
        temp = np.load(path + "tensor_Ntheta={:05d}.npy".format(i))
        sigma_mean[i] = np.mean(temp.flatten())
    print ("Reading data for error model: completed.")

    #2.Plot cantons results  
    fig3, ax3 = plt.subplots(6,5)
    jj = -1
    for result in results:
      jj += 1
      color = colors[jj]
      Name  = Names[jj]
      samples = getPosteriorFromResult(result)
      assert case == 1
      #Plot for individual cantons
      qlo = np.quantile ( a= prediction_matrix[jj], q = 0.5 - p_conf/2 , axis = 0)
      qhi = np.quantile ( a= prediction_matrix[jj], q = 0.5 + p_conf/2 , axis = 0)
      for i0 in range (6):
        for i1 in range (5):
          index = i0 * 5 + i1
          if index <= 25:
            if jj == 1:
              ax3[i0,i1].plot(dates ,exact[index,:],".",color="black",label='reference')

            ax3[i0,i1].text(.5,1.05,name[index],horizontalalignment='center',transform=ax3[i0,i1].transAxes)
            ax3[i0,i1].xaxis.set_major_locator(locator2)  
            ax3[i0,i1].xaxis.set_minor_locator(locator)    
            ax3[i0,i1].xaxis.set_major_formatter(date_form)
            ax3[i0,i1].set(xlabel='', ylabel=r'$I^u$')

            samples_per_canton = 50
            Y = np.zeros((samples.shape[0]//m * samples_per_canton,days))
            tau_real = 2.0
            time   = np.arange(days)
            space  = np.zeros(days)
            aux = np.zeros((days,days))
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
                   aux[i,j] = (sigma_mean[i]*sigma_mean[j])*coef
                else:
                   aux[i,j] = 0.0

            print(index)
            means = prediction_matrix[jj][:,0:days,index]
            for theta in range (samples.shape[0]//m):
                mean = means[theta,:]
                SIG  = samples[theta,7]
                
                Y[theta*samples_per_canton:(theta+1)*samples_per_canton,:] = np.random.multivariate_normal(mean, (SIG*SIG)*aux, samples_per_canton)                
            Y[ np.where(Y<0)] = 0.0
            qlo_ = np.quantile ( a= Y , q = 0.5 - p_conf/2 , axis = 0)
            qhi_ = np.quantile ( a= Y , q = 0.5 + p_conf/2 , axis = 0)
            q50_ = np.quantile ( a= Y , q = 0.5            , axis = 0)
            ax3[i0,i1].fill_between(dates[0:days], qlo_, qhi_,label=Name + ' ' + str(100*p_conf)+"% confidence interval",alpha=0.2,color=color)
            for label in ax3[i0,i1].get_xticklabels():
                label.set_rotation(40)
                label.set_horizontalalignment('right')


    for i0 in range (6):
      for i1 in range (5):
        index = i0 * 5 + i1
        if index > 25:
          fig3.delaxes(ax3[i0][i1])
        else:
          ax3[i0,i1].grid()   

    nnn = "_case" + str(case) + "_sensors" + str(sensors) 
    #handles, labels = ax3[4,1].get_legend_handles_labels()
    #fig3.legend(handles, labels, loc='lower center',ncol=2,bbox_to_anchor=(0.6, 0.1))
    fig3.set_size_inches(14.5, 10.5)
    plt.tight_layout()

    #plt.show()
    fig3.savefig("error.pdf",dpi=1000)



####################################################################################################
if __name__ == "__main__":

  argv = sys.argv[1:]
  parser = argparse.ArgumentParser()
  parser.add_argument('--sensors',type=int)
  parser.add_argument('--case'   ,default=1,type=int)
  args = parser.parse_args(argv)

  sensors=args.sensors
  case=args.case
  assert case == 1

  m=1
  res=[]
  res.append(pickle.load(open("files/optimal_case"+str(case)+"_sensor"+str(sensors)+".pickle", "rb" )))
  res.append(pickle.load(open("files/uniform_case"+str(case)+"_sensor"+str(sensors)+".pickle", "rb" )))

  fname = []
  fname.append("files/optimal_case"+str(case)+"_sensor"+str(sensors)+"_data.npy")
  fname.append("files/uniform_case"+str(case)+"_sensor"+str(sensors)+"_data.npy")
  #if case == 1:
  #  plotNestedResult(res[0],res[1],dims=8,labels=["b\u2080","μ ","α ","Z ","D ","θ ","dispersion","sigma"],fname="marginal_case"+str(case) +"_sensors"+str(sensors))
  #elif case == 2:
  #  plotNestedResult(res[0],res[1],dims=12,labels=["b\u2080","μ ","α ","Z ","D ","θ ","b\u2081","b\u2082","d\u2081","d\u2082","θ\u2081","θ\u2082"],fname="marginal_case"+str(case) +"_sensors"+str(sensors))

  confidence_intervals_CH(results=res,fname=fname,sensors=sensors,m=m,case=case)
