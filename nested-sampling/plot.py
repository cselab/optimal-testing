import numpy as np
import pandas as pd
import pickle
import os,sys
import matplotlib.pyplot as plt
import argparse
from dynesty import plotting as dyplot
from data import *
import datetime
from pandas.plotting import register_matplotlib_converters
from seiin import *
import random
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta

from scipy.stats import nbinom

data      = np.load("canton_daily_cases.npy")
cantons   = data.shape[0] # = 26
days_data = data.shape[1]
name = ['AG','AI','AR','BE','BL','BS','FR','GE','GL','GR','JU','LU','NE',\
        'NW','OW','SG','SH','SO','SZ','TG','TI','UR','VD','VS','ZG','ZH']


####################################################################################################
def resample_equal_with_idx(samples, weights, rstate=None):
####################################################################################################
     if rstate is None:
         rstate = np.random

     if abs(np.sum(weights) - 1.) > 1e-9:  # same tol as in np.random.choice.
         # Guarantee that the weights will sum to 1.
         warnings.warn("Weights do not sum to 1 and have been renormalized.")
         weights = np.array(weights) / np.sum(weights)

     # Make N subdivisions and choose positions with a consistent random offset.
     nsamples = len(weights)
     positions = (rstate.random() + np.arange(nsamples)) / nsamples

     # Resample the data.
     idx = np.zeros(nsamples, dtype=np.int)
     cumulative_sum = np.cumsum(weights)
     i, j = 0, 0
     while i < nsamples:
         if positions[i] < cumulative_sum[j]:
             idx[i] = j
             i += 1
         else:
             j += 1
     return samples[idx], idx

def getPosteriorFromResult(result):
    from dynesty import utils as dyfunc
    weights = np.exp(result.logwt - result.logz[-1]) #normalized weights
    samples = dyfunc.resample_equal(result.samples, weights) #Compute 10%-90% quantiles.
    return samples

def getPosteriorFromResult1(result):
    weights = np.exp(result.logwt - result.logz[-1]) # normalized weights
    samples, idx = resample_equal_with_idx(result.samples, weights)
    return samples, idx

####################################################################################################
def posterior_plots(result,case):
####################################################################################################
    samples, idx  = getPosteriorFromResult1(result)
    numdim     = len(samples[0])
    numentries = len(samples)
    samplesTmp = np.reshape(samples, (numentries, numdim))

    lab= ["b\u2080","μ","α","Z","D"] 
    names = [r'$I^{AG}_u$',
             r'$I^{BE}_u$',
             r'$I^{BL}_u$',
             r'$I^{BS}_u$',
             r'$I^{FR}_u$',
             r'$I^{GE}_u$',
             r'$I^{GR}_u$',
             r'$I^{SG}_u$',
             r'$I^{TI}_u$',
             r'$I^{VD}_u$',
             r'$I^{VS}_u$',
             r'$I^{ZH}_u$',"r"]
    if case == 2:
       lab.append("θ")
       lab.extend(names)
       jmax_1 = 4 
       jmax_2 = 5

    elif case == 3:
       lab2 = ["θ\u2080","b\u2081","b\u2082","d\u2081","d\u2082","θ\u2081","θ\u2082"]
       lab.extend(lab2)
       lab.extend(names)
       jmax_1 = 5 
       jmax_2 = 5

    j_max = int( numdim**0.5 + 1)
    fig,ax = plt.subplots(jmax_1,jmax_2)
    #fig,ax = plt.subplots(jmax_1,jmax_2,sharey=True)
    num_bins = 20
    i = 0
    for j1 in range(jmax_1):
      for j2 in range(jmax_2):
       ax_loc = ax[j1,j2]
       if i >= numdim:
          fig.delaxes(ax_loc)
       else:
          hist, bins, _ = ax_loc.hist(
                           samplesTmp[:, i], num_bins,  color="green", ec='black',alpha=0.5,
                           density=True)
                           #weights=np.zeros_like(samplesTmp[:, i]) + 1. / samplesTmp[:, i].size)
          ax_loc.tick_params(axis='both', which='major', labelsize=4)

          ax_loc.set_title(lab[i],fontsize=8)

       i += 1
    #fig.subplots_adjust(wspace=1., hspace=1.)
    fig.tight_layout()
    fig.savefig("posterior"+str(case)+".pdf",dpi=1000)


####################################################################################################
def model(days,p,case):
####################################################################################################
  if case == 1:
    int_day = 1000
    return example_run_seiin(days,p[0:len(p)-1],int_day)
  else:  
    return example_run_seiin(days,p[0:len(p)-1])



####################################################################################################
def confidence_intervals_daily_reported(result,case,m):
####################################################################################################
    days = days_data + 0#30
    p = 0.99   
    if case == 2:
       days = 21 + 10

    base      = datetime(2020, 2, 25) #February 25th, 2020
    dates     = np.array([base + timedelta(hours=(24 * i)) for i in range(days)])
    dates2    = np.array([base + timedelta(hours=(24 * i)) for i in range(days_data)])
    locator   = mdates.WeekdayLocator(interval=4)
    locator2  = mdates.WeekdayLocator(interval=1)
    formatter = DateFormatter("%b %d")

    samples    = getPosteriorFromResult(result)

    ndim       = samples.shape[1]
    logl       = result.logl
    parameters = samples[np.where( np.abs(logl-np.max(logl))<1e-10 )]
    parameters = parameters.reshape(ndim)
    simulation = model(days,parameters,case)
    np.save("map.npy",parameters)
    print("MAP saved.",flush=True)
    R0 = parameters[0]*parameters[4]*(parameters[1]*(1-parameters[2])+parameters[2])
    mean = np.mean(samples,axis=0)
    print(parameters)
    print(mean)
    np.save("test.npy",mean)
    print("R0=",R0)

    prediction_matrix         = np.zeros ( ( samples.shape[0]//m, days     ) )    
    prediction_matrix_cantons = np.zeros ( ( samples.shape[0]//m, days, 26 ) )    
    for i in range( samples.shape[0]//m):
        print (i,"/",samples.shape[0]//m)
        pp = samples[i,:]
        simulation = model(days=days,p=pp,case=case)
        prediction = []
        for d in range ( days ):
            cases = simulation[d].E() 
            prediction_matrix        [i,d  ] = (pp[2]/pp[3])*np.sum(cases)            
            prediction_matrix_cantons[i,d,:] = (pp[2]/pp[3])*np.asarray(cases)
    np.save("samples.npy",prediction_matrix_cantons)
    np.save("temp.npy",prediction_matrix)
    '''
    prediction_matrix_cantons=np.load("samples.npy")
    prediction_matrix        =np.load("temp.npy")
    '''

    # country
    ################################################################################################
    fig, ax = plt.subplots(constrained_layout=True)
    reference = prepareData(country = True)
    prediction = []
    for i in range ( days ):
        cases = simulation[i].E() 
        prediction.append( parameters[2]/parameters[3]* np.sum(cases) )    
    ax.xaxis.set_major_locator(locator)    
    ax.xaxis.set_major_formatter(formatter)
    ax.plot(dates2,reference ,'-o',label='data'                         ,zorder=10,color="red" )
    ax.plot(dates ,prediction,     label='maximum a posteriori estimate',zorder=5 ,color="blue")    
    q50  = np.quantile ( a= prediction_matrix , q = 0.50      , axis = 0)
    qlo  = np.quantile ( a= prediction_matrix , q = 0.5 - p/2 , axis = 0)
    qhi  = np.quantile ( a= prediction_matrix , q = 0.5 + p/2 , axis = 0)
    plt.plot(dates ,q50       ,label='median prediction',zorder=1,color="black")
    plt.fill_between(dates, qlo, qhi,label=str(100*p)+"% credible interval",color="green")
    fig.legend()
    ax.grid()
    fig.savefig("case" + str(case) + "_prediction_country.pdf",dpi=100 ,format="pdf")
    ################################################################################################





    #Plot cantons
    #############
    fig, axs = plt.subplots(6,5)
    axs.titlesize      : xx-small
    for i0 in range (6):
      for i1 in range (5):
        index = i0 * 5 + i1
        if index > 25:
              fig.delaxes(axs[i0][i1])
        else:
           samples_tot     = samples.shape[0] // m
           samples_per_day = 10
           sam = np.zeros((days,samples_tot*samples_per_day))
           for d in range(days):
              print(d,days,index)
              for s in range(samples_tot):
                mean       = prediction_matrix_cantons[s,d,index]
                dispersion = samples[s,-1] * mean +  1e-10
                pr         = 1.0 / (1.0 + mean/dispersion)
                sam [d,s*samples_per_day:(s+1)*samples_per_day]  = np.random.negative_binomial(n=dispersion, p=pr, size=samples_per_day)

           qlo  = np.quantile ( a= sam , q = 0.5 - p/2 , axis = 1)
           qhi  = np.quantile ( a= sam , q = 0.5 + p/2 , axis = 1)
           axs[i0,i1].fill_between(dates, qlo, qhi,label=str(100*p) + "% credible interval",color="green")
           x=[]
           prediction = [] 
           c_data = []
           for i in range ( days ):
               cases = simulation[i].E() 
               c_data.append( parameters[2]/parameters[3]*cases[index] )
           axs[i0,i1].plot(dates,c_data,label="maximum a posteriori estimate",linewidth=2,color="blue")
           axs[i0,i1].scatter(dates2,data[index,:],s=1.0,label="data",color="red")           
           axs[i0,i1].text(.5,1.05,name[index],horizontalalignment='center',transform=axs[i0,i1].transAxes)
           axs[i0,i1].xaxis.set_major_locator(locator)
           axs[i0,i1].xaxis.set_minor_locator(locator2)
           axs[i0,i1].xaxis.set_major_formatter(formatter)
           axs[i0,i1].set(xlabel='', ylabel='Infections')
           axs[i0,i1].grid()

    handles, labels = axs[4,1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',ncol=2,bbox_to_anchor=(0.6, 0.1))
    fig.set_size_inches(14.5, 10.5)
    plt.tight_layout()
    fig.savefig("case" + str(case) + "_prediction_cantons.pdf",dpi=100 ,format="pdf")
    plt.show()
    print("Done plotting predictions.")



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
if __name__=='__main__':
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--case',type=int,default=2)
    args = parser.parse_args(argv)
    case = args.case
    m = 1

    res = pickle.load( open( "case"+str(case) + "/cantons___"+str(case)+".pickle", "rb" ) )
    res.summary()
    posterior_plots(res,case)
    confidence_intervals_daily_reported(res,case,m)
