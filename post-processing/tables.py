import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from matplotlib.pyplot import figure
from matplotlib import cm
from matplotlib.dates import DateFormatter
import datetime
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters

CANTON_TO_INDEX = {'AG': 0 , 'AI': 1 , 'AR': 2 , 'BE': 3 , 'BL': 4 , 'BS': 5 ,\
                   'FR': 6 , 'GE': 7 , 'GL': 8 , 'GR': 9 , 'JU': 10, 'LU': 11,\
                   'NE': 12, 'NW': 13, 'OW': 14, 'SG': 15, 'SH': 16, 'SO': 17,\
                   'SZ': 18, 'TG': 19, 'TI': 20, 'UR': 21, 'VD': 22, 'VS': 23,\
                   'ZG': 24, 'ZH': 25}
name = ['AG','AI','AR','BE','BL','BS','FR','GE','GL','GR',\
        'JU','LU','NE','NW','OW','SG','SH','SO','SZ','TG',\
        'TI','UR','VD','VS','ZG','ZH']


def table1(v_list,min_day):
  # v_list = utility[ number of sensors ] [ canton ] [ day ]
  sensors = len(v_list)
  cantons = 26
  days    = len(v_list[0][0])
  base    = datetime.datetime(2020, 2, 25) #February 25th, 2020
  dates   = np.array([base + datetime.timedelta(hours=(24 * i)) for i in range(days)])
  date_form = DateFormatter("%b %d")

  print("\\"+"begin{table}[h!]")
  print("\centering")
  print("\\"+"begin{tabular}{|c|c|c|}")
  print("\hline")
  print("Canton & Date & Utility \\\ \hline")
  for n in range (1) : #(sensors):
      t = v_list[n][:][:]
      max_utilities  = np.zeros(cantons,dtype=int)
      max_utilities1 = np.zeros(cantons)
      for c in range (cantons):
          max_utilities [c] = np.argmax(t[c][:])
          max_utilities1[c] = t[c][np.argmax(t[c][:])]
      indices = (-max_utilities1).argsort()
      max_utilities1 = max_utilities1[indices[::-1]]
      max_utilities  = max_utilities [indices[::-1]]
      for c in indices : #range (cantons):
          print(name[c],"&",dates[max_utilities[c]].strftime("%d-%m"),"&","{:6.3f}".format(t[c][max_utilities[c]]),"\\\ \hline  ")
  print("\end{tabular}")
  print("\caption{Optimal measurement days per canton after the outbreak of a new disease.}")
  print("\label{table:case1}")
  print("\end{table}")


def table2(v_list,min_day):
  # v_list = utility[ number of sensors ] [ canton ] [ day ]
  sensors = len(v_list)
  cantons = 26
  days    = len(v_list[0][0])
  base    = datetime.datetime(2020, 2, 25) #February 25th, 2020
  dates   = np.array([base + datetime.timedelta(hours=(24 * i)) for i in range(days)])
  date_form = DateFormatter("%b %d")

  print("\\"+"begin{table}[h!]")
  print("\centering")
  print("\\"+"begin{tabular}{|c|c|c|}")
  print("\hline")
  print("Date &  Proposed Cantons & Estimated Expected Utility  \\\ \hline")
  for d in range(min_day,days):
      t = v_list[0,:,d]
      indices = (-t).argsort()
      i0 = indices[0]
      i1 = indices[1]
      i2 = indices[2]
      print(dates[d].strftime("%d-%m"),"&", name[i0],name[i1],name[i2],"&", "{:6.3f}".format(t[i0]),"{:6.3f}".format(t[i1]),"{:6.3f}".format(t[i2]),"\\\ \hline  ")
  print("\end{tabular}")
  print("\caption{Three highest value of the estimated expected utility at a particular day during the lock-down.}")
  print("\label{table:case2}")
  print("\end{table}")



def table3(v_list,min_day):
  # v_list = utility[ number of sensors ] [ canton ] [ day ]
  sensors = len(v_list)
  cantons = 26
  days    = len(v_list[0][0])
  base    = datetime.datetime(2020, 2, 25) #February 25th, 2020
  dates   = np.array([base + datetime.timedelta(hours=(24 * i)) for i in range(days)])
  date_form = DateFormatter("%b %d")

  print("\\"+"begin{table}[h!]")
  print("\centering")
  print("\\"+"begin{tabular}{|c|c|c|}")
  print("\hline")
  print("Canton & Optimal Day & Estimated Expected Utility \\\ \hline")
  for n in range (1) : #(sensors):
      t = v_list[n][:][:]
      max_utilities  = np.zeros(cantons,dtype=int)
      max_utilities1 = np.zeros(cantons)
      for c in range (cantons):
          max_utilities [c] = np.argmax(t[c][:])
          max_utilities1[c] = t[c][np.argmax(t[c][:])]
      indices = (-max_utilities1).argsort()
      max_utilities1 = max_utilities1[indices[::-1]]
      max_utilities  = max_utilities [indices[::-1]]
      for c in indices : #range (cantons):
          print(name[c],"&",dates[max_utilities[c]].strftime("%B %d"),"&","{:6.3f}".format(t[c][max_utilities[c]]),"\\\ \hline  ")
  print("\end{tabular}")
  print("\caption{Optimal measurement days per canton after loosening of measures.}")
  print("\label{table:case3}")
  print("\end{table}")




#r1 = np.load("result_Ny00800_Nt00800_1.npy")
#table1(r1,0)
#print("")
#r2 = np.load("result_Ny00800_Nt00800_2.npy")
#table2(r2,21)
print("")
r3 = np.load("result_Ny00800_Nt00800_3.npy")
table1(r3,102)
table2(r3,102)
