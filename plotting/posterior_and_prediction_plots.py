import numpy as np
import pandas as pd
import pickle
import os
import sys
import matplotlib.pyplot as plt
import argparse
import time

from dynesty import NestedSampler
from pandas.plotting import scatter_matrix
from dynesty import plotting as dyplot

import datetime
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters


sys.path.append(os.path.join(os.path.dirname(__file__), '../nested-sampling'))
from data import *
from epidemics.cantons.py.run_osp_cases import *

import random

data      = np.load("canton_daily_cases.npy")
cantons   = data.shape[0] # = 26
days_data = data.shape[1]
name = ['AG','AI','AR','BE','BL','BS','FR','GE','GL','GR',\
        'JU','LU','NE','NW','OW','SG','SH','SO','SZ','TG',\
        'TI','UR','VD','VS','ZG','ZH']
    

def getPosteriorFromResult(result):
    from dynesty import utils as dyfunc
    weights = np.exp(result.logwt - result.logz[-1]) #normalized weights
    samples = dyfunc.resample_equal(result.samples, weights) #Compute 10%-90% quantiles.
    return samples



def model(days,p):
  dd = 84
  c = np.random.uniform( 0.0, 1.0, 1)
  slope_max = 0.05
  cc = slope_max * c

  par = [p[0],p[1],p[2],p[3],p[4],p[5],
         p[6],p[7], cc  ,p[8],p[9],dd] 
         #100,100,100,100,100,100] #no interventions up to day 21
  #for i in range (12):
  #    par.append(p[6+i])
  for i in range (12):
      par.append(p[10+i])
  return example_run_seiin(days,par)



####################################################################################################
def confidence_intervals_CH(result,basename,quantile=False,cumulative=False):
####################################################################################################	
    days    = days_data + 60 #plot 60 days after last datapoint
    samples = getPosteriorFromResult(result)
    ndim    = samples.shape[1]

    fig, ax = plt.subplots(constrained_layout=True)

    logl       = result.logl
    parameters = samples[np.where( np.abs(logl-np.max(logl))<1e-10 )]
    parameters = parameters.reshape(ndim)
    simulation = model(days,parameters)

    reference = prepareData(country = True)
    
    prediction = []
    p = 0.90
    for i in range ( days ):
        cases = simulation[i].E() 
        prediction.append( parameters[2]/parameters[3]* np.sum(cases) )
 
    base    = datetime.datetime(2020, 2, 25) #February 25th, 2020
    dates   = np.array([base + datetime.timedelta(hours=(24 * i)) for i in range(days)])
    dates2  = np.array([base + datetime.timedelta(hours=(24 * i)) for i in range(days_data)])

    locator   = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    ax.xaxis.set_major_locator(locator)    
    ax.xaxis.set_major_formatter(formatter)
    if cumulative == True:
      ax.scatter(dates2,np.cumsum(reference ),label="data"                         ,s=1.0   ,color="red" )
      ax.plot   (dates ,np.cumsum(prediction),label='maximum likelihood prediction',zorder=5,color="blue")
    else:
      ax.plot(dates2,reference ,'-o',label='data',zorder=10,color="red")
      ax.plot(dates ,prediction,label='maximum likelihood prediction',zorder=5,color="blue")
 

    if quantile == True and cumulative == False:
       prediction_matrix         = np.zeros ( ( samples.shape[0]//100, days     ) )    
       prediction_matrix_cantons = np.zeros ( ( samples.shape[0]//100, days, 26 ) )    
       for i in range( samples.shape[0]//100 ):
           print (i,"/",samples.shape[0]//100)
           pp = samples[i,:]
           simulation = model(days,pp)
           prediction = []
           for d in range ( days ):
               cases = simulation[d].E() 
               prediction_matrix        [i,d  ] = (pp[2]/pp[3])*np.sum(cases)            
               prediction_matrix_cantons[i,d,:] = (pp[2]/pp[3])*np.asarray(cases)
       np.save(basename + "samples.npy",prediction_matrix_cantons)
       np.save(basename + "temp.npy",prediction_matrix)
       '''
       prediction_matrix_cantons=np.load(basename + "samples.npy")
       prediction_matrix        =np.load(basename + "temp.npy")
       '''
       
       q50  = np.quantile ( a= prediction_matrix , q = 0.50  , axis = 0)
       qlo  = np.quantile ( a= prediction_matrix , q = 0.5 - p/2 , axis = 0)
       qhi  = np.quantile ( a= prediction_matrix , q = 0.5 + p/2 , axis = 0)
       plt.plot(dates ,q50       ,label='median prediction',zorder=1,color="black")
       plt.fill_between(dates, qlo, qhi,label=str(100*p)+"% credible interval",color="green")


    ax.vlines( label = name[7 ] , x=dates [127] , color = "yellow" ,ymin = 0.0 , ymax = 1000 * 2.055 , linestyle="solid",linewidth=2.0) 


    fig.legend()
    ax.grid()
    plt.show()
    if cumulative == True:
      fig.savefig(basename+"prediction_country_cum.pdf",dpi=100 ,format="pdf")
    else:
      fig.savefig(basename+"prediction_country.pdf",dpi=100 ,format="pdf")




    fig, axs   = plt.subplots(6,5)
    locations  = np.arange(1,days+1)

    if quantile == True and cumulative == False:
       q05  = np.quantile ( a= prediction_matrix_cantons , q = 0.5 - p/2 , axis = 0)
       q95  = np.quantile ( a= prediction_matrix_cantons , q = 0.5 + p/2 , axis = 0)

    fig, axs = plt.subplots(6,5)
    axs.titlesize      : xx-small
    for i0 in range (6):
      for i1 in range (5):
        index = i0 * 5 + i1
        if index > 25:
              fig.delaxes(axs[i0][i1])
        else:
           x=[]
           prediction = [] 
           c_data = []
           for i in range ( days ):
               cases = simulation[i].E() 
               c_data.append( parameters[2]/parameters[3]*cases[index] )

           if quantile == True and cumulative == False:
              axs[i0,i1].fill_between(dates, q05[:,index], q95[:,index],label="95% credible interval",color="green")

           if cumulative == True:
              d_ = np.copy(data[index,:])
              nans, x= nan_helper(d_)
              d_[nans]= np.interp(x(nans), x(~nans), d_[~nans])
              #d_[nans]= 0.0              
              axs[i0,i1].plot(dates,np.cumsum(c_data),label="maximum likelihood prediction",linewidth=2,color="blue")
              axs[i0,i1].scatter(dates2, np.cumsum(d_),s=1.0,label="data",color="red")
           else:
              axs[i0,i1].plot(dates,c_data,label="maximum likelihood prediction",linewidth=2,color="blue")
              axs[i0,i1].scatter(dates2,data[index,:],s=1.0,label="data",color="red")
           
           axs[i0,i1].text(.5,1.05,name[index],horizontalalignment='center',transform=axs[i0,i1].transAxes)
           axs[i0,i1].xaxis.set_major_locator(locator)
           axs[i0,i1].xaxis.set_major_formatter(formatter)
           axs[i0,i1].set(xlabel='', ylabel='Infections')
           axs[i0,i1].grid()
           
    handles, labels = axs[4,1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',ncol=2,bbox_to_anchor=(0.6, 0.1))

    fig.set_size_inches(14.5, 10.5)
    plt.tight_layout()
    if cumulative == True:
       fig.savefig(basename+"prediction_cantons_cum.pdf", dpi=100 ,format='pdf')
    else:
       fig.savefig(basename+"prediction_cantons.pdf", dpi=100 ,format='pdf')
    print("Done plotting predictions.")


####################################################################################################
def posterior_plots(result,basename):
####################################################################################################  
    lab = ["b0",
           "μ ",
           "α ",
           "Z ",
           "D ",
           "θ ",
           "b1",
           "b2",
           "d1",
           "d2",
           'AG', #0 
           'BE', #3 
           'BL', #4 
           'BS', #5 
           'FR', #6 
           'GE', #7 
           'GR', #9 
           'SG', #15
           'TI', #20
           'VD', #22
           'VS', #23
           'ZH', #25
           "Dispersion (error model parameter)"]

    ndim = result.samples.shape[1]

    fig, axes = dyplot.traceplot(result, truths=np.zeros(8),
                             truth_color='black', show_titles=True,
                             trace_cmap='viridis', connect=True,smooth=40,
                             connect_highlight=range(5),dims=[0,1,2,3,4,5,6,7],
                             labels=lab[0:8])
    plt.tight_layout()
    fig.savefig(basename+"test1.png")


    fig, axes = dyplot.traceplot(result, truths=np.zeros(8),
                             truth_color='black', show_titles=True,
                             trace_cmap='viridis', connect=True,smooth=40,
                             connect_highlight=range(5),dims=[8,9,10,11,12,13,14,15],
                             labels=lab[8:16])
    plt.tight_layout()
    fig.savefig(basename+"test2.png")


    fig, axes = dyplot.traceplot(result, truths=np.zeros(7),
                             truth_color='black', show_titles=True,
                             trace_cmap='viridis', connect=True,smooth=40,
                             connect_highlight=range(5),dims=[16,17,18,19,20,21,22],
                             labels=lab[16:23])
    plt.tight_layout()
    fig.savefig(basename+"test3.png")







basename = "cantons___"
res = pickle.load( open( basename + ".pickle", "rb" ) )
res.summary()
#posterior_plots(res,basename)
confidence_intervals_CH(result=res,basename=basename,quantile=False,cumulative=False)
#confidence_intervals_CH(result=res,basename=basename,quantile=True,cumulative=False)
#confidence_intervals_CH(result=res,basename=basename,quantile=True,cumulative=True )
