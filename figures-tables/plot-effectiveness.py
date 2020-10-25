import numpy as np
import pickle
import os
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 5})
import argparse,datetime
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from matplotlib.dates import DateFormatter
from datetime import timedelta

import matplotlib.colors as mcolors
COLORS = mcolors.CSS4_COLORS

ic_cantons=12
cantons = 26
p_conf = 0.99
name = ['AG','AI','AR','BE','BL','BS','FR','GE','GL','GR','JU','LU','NE',\
        'NW','OW','SG','SH','SO','SZ','TG','TI','UR','VD','VS','ZG','ZH']

def distance(t1,t2,tau):
    dt = np.abs(t1-t2) / tau
    return np.exp(-dt)

def getPosteriorFromResult(result):
    from dynesty import utils as dyfunc
    weights = np.exp(result.logwt - result.logz[-1]) #normalized weights
    samples = dyfunc.resample_equal(result.samples, weights) #Compute 10%-90% quantiles.
    return samples

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
####################################################################################################
def getPosteriorFromResult1(result):
#################################################################################################### 
    weights = np.exp(result.logwt - result.logz[-1]) # normalized weights
    samples, idx = resample_equal_with_idx(result.samples, weights)
    return samples, idx
####################################################################################################
def plot_histogram(ax, theta,dims,color,alpha):
####################################################################################################
  dim = dims 
  num_bins = 30
  for i in range(dim):
    if (dim == 1):
      ax_loc = ax
    else:
      ax_loc = ax[i, i]

    hist, bins, _ = ax_loc.hist(
        theta[:, i], num_bins,  color=color, ec='black',alpha=alpha,
        density=True, lw=0)
        #weights=np.zeros_like(theta[:, i]) + 1. / theta[:, i].size)
####################################################################################################
def plot_lower_triangle(ax, theta,dims,hide):
####################################################################################################
  dim = dims #theta.shape[1]
  if (dim == 1):
    return

  for i in range(dim):
    for j in range(i):
      # returns bin values, bin edges and bin edges
      H, xe, ye = np.histogram2d(theta[:, j], theta[:, i], 10, density=True)
      # plot and interpolate data
      ax[i, j].imshow(
          H.T,
          aspect="auto",
          interpolation='spline16',
          origin='lower',
          extent=np.hstack((ax[j, j].get_xlim(), ax[i, i].get_xlim())),
          cmap=plt.get_cmap('inferno'))
####################################################################################################
def plotNestedResult(result1,result2,dims,labels,fname):
####################################################################################################
    fig, ax = plt.subplots(dims, dims)

    samples, idx  = getPosteriorFromResult1(result1)
    numdim     = len(samples[0])
    numentries = len(samples)
    samplesTmp = np.reshape(samples, (numentries, numdim))
    plot_histogram(ax, samplesTmp,dims,color=COLORS['mediumorchid'],alpha=0.5)
    plot_lower_triangle(ax, samplesTmp,dims,False)

    samples, idx  = getPosteriorFromResult1(result2)
    numdim     = len(samples[0])
    numentries = len(samples)
    samplesTmp = np.reshape(samples, (numentries, numdim))
    plot_histogram(ax, samplesTmp,dims,color=COLORS['silver'],alpha=0.5)
    plot_lower_triangle(ax.T, samplesTmp,dims,True)

    for i in range (dims):
       for j in range (dims):
          if i!=j:
             if i<j:
                ax[i,j].set_xlabel(labels[i])
                ax[i,j].set_ylabel(labels[j])
             else:
                ax[i,j].set_xlabel(labels[j])
                ax[i,j].set_ylabel(labels[i])
          else:
             ax[i,j].set_xlabel(labels[i])
             #ax[i,j].set_ylabel("relative frequency")
             ax[i,j].set_ylabel("")

    #plt.tight_layout()
    # plt.subplots_adjust(wspace=0.5, hspace=0.5)
    fig.set_size_inches(7, 6.5)
    fig.tight_layout()
    
    plt.savefig(fname+".pdf")#,format="pdf")#,dpi=100,bbox_inches='tight')


####################################################################################################
def confidence_intervals_CH(results,fname,sensors,m=1,case=1):
####################################################################################################	
    days = 8 + 6
    base  = datetime.datetime(2020, 2, 25) #February 25th, 2020
    dates = np.array([base + datetime.timedelta(hours=(24 * i)) for i in range(days)])
    locator   = mdates.DayLocator(interval=1)
    locator2  = mdates.WeekdayLocator(interval=1)
    date_form = DateFormatter("%b %d")

    exact = np.load("exact.npy")

    #2. Plot country and cantons results  
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(locator)    
    ax.xaxis.set_major_formatter(date_form)
    ax.plot(dates ,np.sum(exact,axis=0),".",color="black",label='reference', markersize=1)
    for label in ax.get_xticklabels():
      label.set_rotation(40)
      label.set_horizontalalignment('right')
    ax.set_xlabel("Days (t)")
    ax.set_ylabel("Unreported Infectious")

    #3. Run model for all samples of theta
    colors = [COLORS['mediumorchid'],COLORS['silver']]
    Names = ['optimal testing','sub-optimal testing']
    prediction_matrix = []  
    jj = -1
    for result in results:
        jj += 1
        a = np.load("model_results"+str(jj)+".npy")
        prediction_matrix.append(a)

    fig2, axs   = plt.subplots(6,5)
    jj = -1
    for result in results:
      jj += 1
      Name = Names[jj]
      color = colors[jj]
      alpha = 0.2
      samples = getPosteriorFromResult(result)
      
      #Plot results for all of Switzerland
      q50  = np.quantile ( a= np.sum(prediction_matrix[jj],axis=2) , q = 0.5            , axis = 0)
      qlo  = np.quantile ( a= np.sum(prediction_matrix[jj],axis=2) , q = 0.5 - p_conf/2 , axis = 0)
      qhi  = np.quantile ( a= np.sum(prediction_matrix[jj],axis=2) , q = 0.5 + p_conf/2 , axis = 0)
      ax.fill_between(dates, qlo, qhi,label=Name + ' ' + str(100*p_conf)+"% credible interval",alpha=alpha,color=color)

      #Plot for individual cantons
      qlo = np.quantile ( a= prediction_matrix[jj], q = 0.5 - p_conf/2 , axis = 0)
      qhi = np.quantile ( a= prediction_matrix[jj], q = 0.5 + p_conf/2 , axis = 0)
      q50 = np.quantile ( a= prediction_matrix[jj], q = 0.5            , axis = 0)
      for i0 in range (6):
        for i1 in range (5):
          index = i0 * 5 + i1
          if index <= 25:
            axs[i0,i1].fill_between(dates, qlo[:,index], qhi[:,index],label=Name + ' ' + str(100*p_conf)+"% credible interval",alpha=alpha,color=color,zorder=10-jj)
            axs[i0,i1].text(.5,1.05,name[index],horizontalalignment='center',transform=axs[i0,i1].transAxes)
            axs[i0,i1].xaxis.set_major_locator(locator)  
            axs[i0,i1].xaxis.set_minor_locator(locator2)    
            axs[i0,i1].xaxis.set_major_formatter(date_form)
            for label in axs[i0,i1].get_xticklabels():
              label.set_rotation(40)
              label.set_horizontalalignment('right')
            if jj == 1:
              axs[i0,i1].plot(dates ,exact[index,:],".",color="black",label='reference', markersize=1, zorder=11)
            data = np.load(fname[jj])
            points = int( len(data)/3 )
            day_vector    = data[0*points:1*points]
            canton_vector = data[1*points:2*points]
            case_vector   = data[2*points:3*points]

    for i0 in range (6):
      for i1 in range (5):
        index = i0 * 5 + i1
        if index > 25:
          fig2.delaxes(axs[i0][i1])

    handles, labels = axs[4,1].get_legend_handles_labels()
    fig2.legend(handles, labels, loc='lower center',ncol=2,bbox_to_anchor=(0.6, 0.1))
    fig2.set_size_inches(14.5, 10.5)
    fig2.tight_layout()

    nnn = "_case" + str(case) + "_sensors" + str(sensors) 

    fig.set_size_inches(3.42, 1.5)
    fig.tight_layout()
    fig.savefig ("Iu_country" +nnn+ ".pdf") #,dpi=1000)
    fig2.savefig("Iu_cantons" +nnn+ ".png",dpi=1000)


    print("Plotting error + model confidence intervals... (this may take a while)")
    sigma_mean = np.load("sigma_mean.npy")
    fig3, ax3 = plt.subplots(6,5)
    jj = -1
    for result in results:
      jj += 1
      color = colors[jj]
      Name  = Names[jj]
      samples = getPosteriorFromResult(result)

      #Plot for individual cantons
      qlo = np.quantile ( a= prediction_matrix[jj], q = 0.5 - p_conf/2 , axis = 0)
      qhi = np.quantile ( a= prediction_matrix[jj], q = 0.5 + p_conf/2 , axis = 0)
      q50 = np.quantile ( a= prediction_matrix[jj], q = 0.5            , axis = 0)
      for i0 in range (6):
        for i1 in range (5):
          index = i0 * 5 + i1
          if index <= 25:
            if jj == 1:
              ax3[i0,i1].plot(dates ,exact[index,:],".",label='reference',color="black", markersize=1)

            ax3[i0,i1].text(.5,1.05,name[index],horizontalalignment='center',transform=ax3[i0,i1].transAxes)
            ax3[i0,i1].xaxis.set_major_locator(locator)  
            ax3[i0,i1].xaxis.set_minor_locator(locator2)    
            ax3[i0,i1].xaxis.set_major_formatter(date_form)
            for label in ax3[i0,i1].get_xticklabels():
              label.set_rotation(40)
              label.set_horizontalalignment('right')

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

            means = prediction_matrix[jj][:,0:days,index]
            for theta in range (samples.shape[0]//m):
                mean = means[theta,:]
                SIG  = samples[theta,7]
                
                Y[theta*samples_per_canton:(theta+1)*samples_per_canton,:] = np.random.multivariate_normal(mean, (SIG*SIG)*aux, samples_per_canton)                
            Y[ np.where(Y<0)] = 0.0
            qlo_ = np.quantile ( a= Y , q = 0.5 - p_conf/2 , axis = 0)
            qhi_ = np.quantile ( a= Y , q = 0.5 + p_conf/2 , axis = 0)
            q50_ = np.quantile ( a= Y , q = 0.5            , axis = 0)
            ax3[i0,i1].fill_between(dates[0:days], qlo_, qhi_,label=Name + ' ' + str(100*p_conf)+"% credible interval",alpha=0.4,color=color)

    for i0 in range (6):
      for i1 in range (5):
        index = i0 * 5 + i1
        if index > 25:
          fig3.delaxes(ax3[i0][i1])

    nnn = "_case" + str(case) + "_sensors" + str(sensors) 
    fig3.set_size_inches(14.5, 10.5)
    plt.tight_layout()
    fig3.savefig("Iu_cantons_error" +nnn+ ".png",dpi=100 )



####################################################################################################
if __name__ == "__main__":
  case = 1 
  sensors = 2 

  os.chdir("./effectiveness")

  m=1
  res=[]
  res.append(pickle.load(open("optimal_case"+str(case)+"_sensor"+str(sensors)+".pickle", "rb" )))
  res.append(pickle.load(open("uniform_case"+str(case)+"_sensor"+str(sensors)+".pickle", "rb" )))
  fname = []
  fname.append("optimal_case"+str(case)+"_sensor"+str(sensors)+"_data.npy")
  fname.append("uniform_case"+str(case)+"_sensor"+str(sensors)+"_data.npy")

  confidence_intervals_CH(results=res,fname=fname,sensors=sensors,m=m,case=case)
  if case == 1:
    plotNestedResult(res[0],res[1],dims=8,labels=["b\u2080","μ ","α ","Z ","D ","θ ","r","c"],fname="marginal_case"+str(case) +"_sensors"+str(sensors))
