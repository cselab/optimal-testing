#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import os
import sys
from plot import Renderer
from matplotlib.pyplot import figure
import datetime
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters

name = ['AG','AI','AR','BE','BL','BS','FR','GE','GL','GR',\
        'JU','LU','NE','NW','OW','SG','SH','SO','SZ','TG',\
        'TI','UR','VD','VS','ZG','ZH']

CANTON_TO_INDEX = {'AG': 0 , 'AI': 1 , 'AR': 2 , 'BE': 3 , 'BL': 4 , 'BS': 5 ,\
                   'FR': 6 , 'GE': 7 , 'GL': 8 , 'GR': 9 , 'JU': 10, 'LU': 11,\
                   'NE': 12, 'NW': 13, 'OW': 14, 'SG': 15, 'SH': 16, 'SO': 17,\
                   'SZ': 18, 'TG': 19, 'TI': 20, 'UR': 21, 'VD': 22, 'VS': 23,\
                   'ZG': 24, 'ZH': 25}
NUM_CANTONS = len(CANTON_TO_INDEX)



def plot_all_cantons(v_list):
  # v_list = utility[ number of sensors ] [ canton ] [ day ]
  sensors = len(v_list)
  cantons = 26
  days    = len(v_list[0][0])

  v = np.zeros( (sensors,cantons*days) )
  for i in range(sensors):
    for j in range(cantons):
      for k in range(days  ):
        v[i][ j*days + k ] = v_list[i][j][k]

  print (days)
  max_v = np.max(v_list)
  
  locations_y = np.arange(0, int(max_v+1),2)
  base    = datetime.datetime(2020, 2, 25) #February 25th, 2020
  dates   = np.array([base + datetime.timedelta(hours=(24 * i)) for i in range(days)])
  locator = mdates.DayLocator(interval=1)
  locator2 = mdates.WeekdayLocator(interval=2)
  formatter = mdates.ConciseDateFormatter(locator)
  date_form = DateFormatter("%b %d")

  fig, axs = plt.subplots(6,5)
  axs.titlesize      : xx-small
  for i0 in range (6):
    for i1 in range (5):
      index = i0 * 5 + i1
      print(i0,i1)
      if index > 25:
        fig.delaxes(axs[i0][i1])
      else:
        for s in range(sensors):
          lab = str(s+1) + " sensors "
          if s == 0:
            lab = str(s+1) + " sensor "
          axs[i0,i1].plot(dates,v_list[s][index][:],label=lab)
        axs[i0,i1].grid()
        axs[i0,i1].set_ylim([0.0,max_v])


        axs[i0,i1].xaxis.set_major_locator(locator2)    
        axs[i0,i1].xaxis.set_minor_locator(locator)    
        axs[i0,i1].xaxis.set_major_formatter(date_form)

        axs[i0,i1].text(.5,1.05,name[index],
            horizontalalignment='center',
            transform=axs[i0,i1].transAxes)
        axs[i0,i1].set_yticks(locations_y)

        axs[i0,i1].set(xlabel='Day', ylabel='Utility')
        axs[i0,i1].set_xlim(dates[84],dates[-1])

  
  handles, labels = axs[4,1].get_legend_handles_labels()
  fig.legend(handles, labels, loc='lower center',ncol=sensors,bbox_to_anchor=(0.6, 0.1))
    
  fig.set_size_inches(15., 15.)
  plt.tight_layout()
  fig.savefig('slice.pdf', dpi=100 ,format='pdf')



#################################
def make_movie(result,utility,n):
#################################
    """
    v = utility[number of sensors][canton][day]
    """
    days = utility.shape[2]

    def frame_callback(rend):
        t = rend.get_frame() * (days - 1) // rend.get_max_frame()
        util = utility[n,:,t]
        res  = result[t,:]
        max_res = np.max(result)
        
        v_u = {}
        v_r1 = {}
        v_r2 = {}
        texts = {}
        for i, c in enumerate(rend.get_codes()):
            i_state = CANTON_TO_INDEX[c]
            v_u[c] = util[i_state]
            v_r1[c] = res [i_state]/max_res
            v_r2[c] = res [i_state]
            tt = (np.asarray(v_r2[c])).astype(int)
            texts[c] = tt.astype(str)

        rend.set_values(v_r1)
        rend.set_texts(texts)
        rend.set_bars(v_u)
        plt.suptitle("Day :" + str(t), fontsize=12)
    rend = Renderer(frame_callback)
    rend.save_movie(frames=days,filename=str(n) + ".mp4")


##########################
if __name__ == '__main__':
##########################  
  parser = argparse.ArgumentParser()

  parser.add_argument('--result',help='utility result.npy file'          ,type=str)
  parser.add_argument('--output',help='model evaluations output.npy file',type=str)
  parser.add_argument('--movie' ,help='make movie or not (1 or 0)'       ,type=int)

  args = vars(parser.parse_args())

  print(args["result"])
  print(args["output"])

  utility = np.load(args["result"])


  plot_all_cantons(utility)
  
  #this fails on Euler!
  if args["movie"] == 1:
     results = np.load(args["output"])
     #res = results[0,:,:]
     res = results
     for n in range(0,utility.shape[0]):
         make_movie(res,utility,n)
