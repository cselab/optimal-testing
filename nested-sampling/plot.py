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
sys.path.append(os.path.join(os.path.dirname(__file__), '../../covid19/epidemics/cantons/py'))
from run_osp_cases import *
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
def plot_second_wave(result):
####################################################################################################
    days      = days_data + 60
    samples   = getPosteriorFromResult(result)
    ndim      = samples.shape[1]
    reference = prepareData(country = True)

    fig, ax = plt.subplots(constrained_layout=True)
    
    base   = datetime(2020, 2, 25) #February 25th, 2020
    dates  = np.array([base + timedelta(hours=(24 * i)) for i in range(days)])
    dates2 = np.array([base + timedelta(hours=(24 * i)) for i in range(days_data)])

    ax.xaxis.set_major_formatter(DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    
    ax.plot(dates2,reference,'-o',label='data',zorder=10,color="red")
 
    #prediction_matrix = np.zeros ( ( samples.shape[0]//10, days     ) )    
    #for i in range( samples.shape[0]//10 ):
    #    print (i,"/",samples.shape[0]//10)
    #    pp = samples[i,:]
    #    simulation = model(days=days,p=pp,case=3)
    #    prediction = []
    #    for d in range ( days ):
    #        cases = simulation[d].E() 
    #        prediction_matrix        [i,d  ] = (pp[2]/pp[3])*np.sum(cases)            
    #np.save("temp.npy",prediction_matrix)
    prediction_matrix  = np.load("temp.npy")
       
    p = 0.99
    q50  = np.quantile ( a= prediction_matrix , q = 0.50  , axis = 0)
    qlo  = np.quantile ( a= prediction_matrix , q = 0.5 - p/2 , axis = 0)
    qhi  = np.quantile ( a= prediction_matrix , q = 0.5 + p/2 , axis = 0)

    for i in range(prediction_matrix.shape[0]):
     plt.plot(dates,prediction_matrix[i,:],zorder=1,color="blue",linewidth=0.02)

    plt.fill_between(dates, qlo, qhi,label=str(100*p)+"% credible interval",color="green")

    #t_opt   = [118       ,143        , 143       , 113       , 125       , 130       ,   118,109,143,125,143,122,124,143,143,121,143,127,138,128,112,143,109,114,139,112]
    
    t_opt = [ 97,139,131,94,102,106,97,94,139,101,118,100,102,137,138,99,121,103,109,105,94,139,94,95,112,94]
    utility = [1.68513504 ,0.14490158 ,0.37947703 ,2.25234775 ,1.1366983  ,0.92704742 ,1.54341902 ,2.23957092 ,0.30955075 ,1.10754327 ,0.54688461 ,1.30160695 ,1.10305379 ,0.32835869 ,0.3006956  ,1.43490351 ,0.49342578 ,1.12727342 ,0.76470072 ,1.03804126 ,2.06128099 ,0.29061078 ,2.53824248 ,1.78931474 ,0.67347802 ,2.35477999 ]

    utility = ( np.asarray(utility) ) 



    ax.set_ylabel("Daily reported infections")
    ax.set_ylim([0,1500])
    ax.set_xlim([mdates.date2num(dates[0]),mdates.date2num(dates[120])])


    a = np.asarray(utility).argsort()[-10:][::-1]
    col = ['brown','teal','crimson','yellow','orange','green','pink','lightblue','magenta','purple']
    for j in range(10):
        c = a[j]
        d_opt = dates[t_opt[c]]
        start = mdates.date2num(d_opt - timedelta(hours=10) )
        end   = mdates.date2num(d_opt + timedelta(hours=10) )
        width = end-start
        rect = Rectangle( ( start,0 ), width, utility[c]*500,color=col[j],zorder=1000+10*j,alpha=0.5)
        ax.add_patch(rect) 
        height = rect.get_height()

        dx = 0.0
        if name[c] == 'BE':
        	dx = 1.5
        elif name[c] == 'GE':
        	dx = -1.5

        print (name[c],utility[c],t_opt[c]) #,d_opt[c])

        ax.text(rect.get_x() + rect.get_width() / 2 + dx, height + 5, name[c],ha='center', va='bottom',zorder=1000+10*j)

    #fig.legend(loc='upper left')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15),
              ncol=2, fancybox=True, shadow=True)


    ax.grid()
    plt.tight_layout()
    plt.show()
    fig.savefig("second_wave.pdf",dpi=100 ,format="pdf")
    print("Second wave plotted successfully.")



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
    parser.add_argument('--case'      ,type=int,default=2)
    parser.add_argument('--wave'      ,type=int,default=0)
    parser.add_argument('--scenarios' ,type=int,default=0)
    args = parser.parse_args(argv)
    case = args.case
    wave = args.wave
    m = 1
    if case > 0:

        res = pickle.load( open( "case"+str(case) + "/cantons___"+str(case)+".pickle", "rb" ) )
        #res = pickle.load( open( "case"+str(case) + "/cantons___"+str(case)+".pickle", "rb" ) )
        #res = pickle.load( open( "case"+str(case) + "/cantons___"+str(case)+".pickle", "rb" ) )
        res.summary()
        posterior_plots(res,case)
        confidence_intervals_daily_reported(res,case,m)
        #if case == 3 and wave == 1:
        #   plot_second_wave(res)
