import numpy as np
import matplotlib.pyplot as plt
import pickle
from dynesty import plotting as dyplot

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
        density=True)
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
          cmap=plt.get_cmap('jet'))
####################################################################################################
def plotNestedResult(result1,result2,dims,labels,fname):
####################################################################################################
    fig, ax = plt.subplots(dims, dims, figsize=(20, 20))

    samples, idx  = getPosteriorFromResult1(result1)
    numdim     = len(samples[0])
    numentries = len(samples)
    samplesTmp = np.reshape(samples, (numentries, numdim))
    plot_histogram(ax, samplesTmp,dims,color='lightgreen',alpha=0.5)
    plot_lower_triangle(ax, samplesTmp,dims,False)


    samples, idx  = getPosteriorFromResult1(result2)
    numdim     = len(samples[0])
    numentries = len(samples)
    samplesTmp = np.reshape(samples, (numentries, numdim))
    plot_histogram(ax, samplesTmp,dims,color='red',alpha=0.5)
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
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    
    plt.savefig(fname+".pdf",format="pdf",dpi=100,bbox_inches='tight')
