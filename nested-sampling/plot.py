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
sys.path.append(os.path.join(os.path.dirname(__file__), '../covid19/epidemics/cantons/py'))
from run_osp_cases import *
import random
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta

data      = np.load("canton_daily_cases.npy")
cantons   = data.shape[0] # = 26
days_data = data.shape[1]
name = ['AG','AI','AR','BE','BL','BS','FR','GE','GL','GR','JU','LU','NE',\
        'NW','OW','SG','SH','SO','SZ','TG','TI','UR','VD','VS','ZG','ZH']
####################################################################################################
def getPosteriorFromResult(result):
####################################################################################################
    from dynesty import utils as dyfunc
    weights = np.exp(result.logwt - result.logz[-1]) #normalized weights
    samples = dyfunc.resample_equal(result.samples, weights) #Compute 10%-90% quantiles.
    return samples
####################################################################################################
def posterior_plots(result,case):
####################################################################################################
    print("Plotting posterior distributions...")

    names = ['AG','BE','BL','BS','FR','GE','GR','SG','TI','VD','VS','ZH'] 

    lab = ["b\u2080","μ ","α ","Z ","D ","θ "]

    if case == 3:
      lab.append("b\u2081")
      lab.append("b\u2082")
      lab.append("d\u2081")
      lab.append("d\u2082")

    lab.extend(names)
    lab.append("Dispersion (error model parameter)")

    ndim = len(lab)

    if case == 3:
        fig, axes = dyplot.traceplot(result, truths=np.zeros(8),
                                 truth_color='black', show_titles=True,
                                 trace_cmap='viridis', connect=True,smooth=40,
                                 dims=[0,1,2,3,4,5,6,7],labels=lab[0:8])
        for i in range(8):
          fig.delaxes(axes[i,0])
        plt.tight_layout()
        fig.savefig("case3_posterior1.png")


        fig, axes = dyplot.traceplot(result, truths=np.zeros(8),
                                 truth_color='black', show_titles=True,
                                 trace_cmap='viridis', connect=True,smooth=40,
                                 dims=[8,9,10,11,12,13,14,15],labels=lab[8:16])
        for i in range(8):
          axes[i,0].set_visible(False)
        plt.tight_layout()
        fig.savefig("case3_posterior2.png")


        fig, axes = dyplot.traceplot(result, truths=np.zeros(7),
                                 truth_color='black', show_titles=True,
                                 trace_cmap='viridis', connect=True,smooth=40,
                                 dims=[16,17,18,19,20,21,22],labels=lab[16:23])
        for i in range(7):
          axes[i,0].set_visible(False)
        plt.tight_layout()
        fig.savefig("case3_posterior3.png")


    elif case == 2:
        fig, axes = dyplot.traceplot(result, truths=np.zeros(6),
                                 truth_color='black', show_titles=True,
                                 connect=True,smooth=40,
                                 dims=[0,1,2,3,4,5],labels=lab[0:6])
        plt.tight_layout()
        fig.savefig("case2_posterior1.png")

        fig, axes = dyplot.traceplot(result, truths=np.zeros(6),
                                 truth_color='black', show_titles=True,
                                 connect=True,smooth=40,
                                 dims=[6,7,8,9,10,11],labels=lab[6:12])
        plt.tight_layout()
        fig.savefig("case2_posterior2.png")

        fig, axes = dyplot.traceplot(result, truths=np.zeros(6),
                                 truth_color='black', show_titles=True,
                                 connect=True,smooth=40,
                                 dims=[12,13,14,15,16,17],labels=lab[12:18])
        plt.tight_layout()
        fig.savefig("case2_posterior3.png")



        fig, axes = dyplot.traceplot(result, truths=np.zeros(7),
                                 truth_color='black', show_titles=True,
                                 connect=True,smooth=40,
                                 dims=[12,13,14,15,16,17,18],labels=lab[12:19])
        plt.tight_layout()
        fig.savefig("case2_posterior4.png")


    print("Posterior distributions plotted successfully.")
####################################################################################################
def model(days,p,case):
####################################################################################################
  if case == 3:
    d_data = 84
    slope_max = 0.5
    cc = slope_max * np.random.uniform(0.0, 1.0, 1)
    par = [p[0],p[1],p[2],p[3],p[4],p[5],
           p[6],p[7],cc,
           p[8],p[9],d_data] 
    for i in range (12):
        par.append(p[10+i])
    return example_run_seiin(days,par)
  elif case == 2:
    par = [p[0],p[1],p[2],p[3],p[4],p[5],
           p[0],p[0],p[0],
           days,days,days] 
    for i in range (12):
        par.append(p[6+i])
    return example_run_seiin(days,par)
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
def confidence_intervals_CH(result,case,cumulative=False):
####################################################################################################
    days       = days_data + 20
    if case == 2:
       days = 21 + 10
    samples    = getPosteriorFromResult(result)



    print("samples=",samples.shape)
    ndim       = samples.shape[1]

    fig, ax = plt.subplots(constrained_layout=True)

    logl       = result.logl
    parameters = samples[np.where( np.abs(logl-np.max(logl))<1e-10 )]
    parameters = parameters.reshape(ndim)
    simulation = model(days,parameters,case)
    np.save("mle.npy",parameters)

    reference = prepareData(country = True)
    
    prediction = []
    for i in range ( days ):
        cases = simulation[i].E() 
        prediction.append( parameters[2]/parameters[3]* np.sum(cases) )
 
    base    = datetime(2020, 2, 25) #February 25th, 2020
    dates   = np.array([base + timedelta(hours=(24 * i)) for i in range(days)])
    dates2  = np.array([base + timedelta(hours=(24 * i)) for i in range(days_data)])

    locator   = mdates.WeekdayLocator(interval=4)
    locator2  = mdates.WeekdayLocator(interval=1)
    formatter = DateFormatter("%b %d")
    
    ax.xaxis.set_major_locator(locator)    
    ax.xaxis.set_major_formatter(formatter)

    if cumulative == True:
      ax.scatter(dates2,np.cumsum(reference ),label="data",s=1.0,color="red" )
      ax.plot   (dates ,np.cumsum(prediction),label='maximum a posteriori estimate',zorder=5,color="blue")
    else:
      ax.plot(dates2,reference ,'-o',label='data',zorder=10,color="red")
      ax.plot(dates ,prediction,label='maximum a posteriori estimate',zorder=5,color="blue")
 
    prediction_matrix         = np.zeros ( ( samples.shape[0], days     ) )    
    prediction_matrix_cantons = np.zeros ( ( samples.shape[0], days, 26 ) )    
    for i in range( samples.shape[0]):
        print (i,"/",samples.shape[0])
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
    
    p = 0.99   
    q50  = np.quantile ( a= prediction_matrix , q = 0.50  , axis = 0)
    qlo  = np.quantile ( a= prediction_matrix , q = 0.5 - p/2 , axis = 0)
    qhi  = np.quantile ( a= prediction_matrix , q = 0.5 + p/2 , axis = 0)
    plt.plot(dates ,q50       ,label='median prediction',zorder=1,color="black")
    plt.fill_between(dates, qlo, qhi,label=str(100*p)+"% credible interval",color="green")

    fig.legend()
    ax.grid()
    if cumulative == True:
      fig.savefig("case" + str(case) + "_prediction_country_cum.pdf",dpi=100 ,format="pdf")
    else:
      fig.savefig("case" + str(case) + "_prediction_country.pdf",dpi=100 ,format="pdf")


    fig, axs   = plt.subplots(6,5)

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

           axs[i0,i1].fill_between(dates, q05[:,index], q95[:,index],label="95% credible interval",color="green")

           if cumulative == True:
              d_ = np.copy(data[index,:])
              nans, x= nan_helper(d_)
              d_[nans]= np.interp(x(nans), x(~nans), d_[~nans])
              #d_[nans]= 0.0              
              axs[i0,i1].plot(dates,np.cumsum(c_data),label="maximum a posteriori estimate",linewidth=2,color="blue")
              axs[i0,i1].scatter(dates2, np.cumsum(d_),s=1.0,label="data",color="red")
           else:
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
    if cumulative == True:
      fig.savefig("case" + str(case) + "_prediction_cantons_cum.pdf",dpi=100 ,format="pdf")
    else:
      fig.savefig("case" + str(case) + "_prediction_cantons.pdf",dpi=100 ,format="pdf")
    print("Done plotting predictions.")
####################################################################################################
def plot_scenarios():
####################################################################################################
    days       = days_data 
    fig, ax = plt.subplots(constrained_layout=True)
    reference = prepareData(country = True)
    prediction = []
    base    = datetime.datetime(2020, 2, 25) #February 25th, 2020
    dates   = np.array([base + datetime.timedelta(hours=(24 * i)) for i in range(days)])
    dates2  = np.array([base + datetime.timedelta(hours=(24 * i)) for i in range(days+30)])

    locator = mdates.DayLocator(interval=1)
    locator2 = mdates.WeekdayLocator(interval=2)
    formatter = mdates.ConciseDateFormatter(locator)
    date_form = DateFormatter("%b %d")
    ax.xaxis.set_major_locator(locator2)    
    ax.xaxis.set_minor_locator(locator)    
    ax.xaxis.set_major_formatter(date_form)
    ax.axvspan(dates[0], dates[21], alpha=0.4, color='red')
    ax.axvspan(dates[21], dates[-1], alpha=0.4, color='green')
    ax.axvspan(dates2[-31], dates2[-1], alpha=0.4, color='blue')

    ax.bar(dates,reference, width=0.6,label='Daily reported cases',color="black")
    fig.legend()
    plt.show()

    #fig.set_size_inches(14.5, 10.5)
    fig.savefig("scenarios.pdf",dpi=100 ,format="pdf")


if __name__=='__main__':
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--case'      ,type=int,default=2)
    parser.add_argument('--wave'      ,type=int,default=0)
    parser.add_argument('--cumulative',type=int,default=0)
    parser.add_argument('--scenarios' ,type=int,default=0)
    args = parser.parse_args(argv)
    case = args.case
    cum  = args.cumulative
    wave = args.wave
    if case > 0:
        base = "../results/case" + str(case)
        res = pickle.load( open( base + "/cantons___.pickle", "rb" ) )
        res.summary()
        #posterior_plots(res,case)
        #confidence_intervals_CH(res,case,cum)
        if case == 3 and wave == 1:
        	plot_second_wave(res)
