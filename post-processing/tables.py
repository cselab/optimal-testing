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


def table1(v_list,min_day,caption,case):
  # v_list = utility[ number of sensors ] [ canton ] [ day ]
  sensors = len(v_list)
  cantons = 26
  days    = len(v_list[0][0])
  base    = datetime.datetime(2020, 2, 25) #February 25th, 2020
  dates   = np.array([base + datetime.timedelta(hours=(24 * i)) for i in range(days)])
  date_form = DateFormatter("%b %d")

  print("\\"+"begin{table}[H]")
  print("\centering")
  print("\\"+"begin{tabular}{|c|c||c||c||c|}")
  print("\hline")
  print("Canton & 1\\textsuperscript{st} measurement & 2\\textsuperscript{nd} measurement & 3\\textsuperscript{rd} measurement & 4\\textsuperscript{th} measurement \\\ \hline")
  v_list[1][:][:] -= np.max(v_list[0][:][:])
  v_list[2][:][:] -= (np.max(v_list[1][:][:]) + np.max(v_list[0][:][:]))
  v_list[3][:][:] -= (np.max(v_list[2][:][:]) + np.max(v_list[1][:][:]) + np.max(v_list[0][:][:]))
  for c in range (cantons):
      t0 = v_list[0][c][:]
      t1 = v_list[1][c][:] 
      t2 = v_list[2][c][:] 
      t3 = v_list[3][c][:]
      d0 = np.argmax(t0)
      d1 = np.argmax(t1)
      d2 = np.argmax(t2)
      d3 = np.argmax(t3)
      print(name[c],"&",
                        "{:6.3f}".format(t0[d0]),dates[d0].strftime("(%d-%m)"),"&",
                        "{:6.3f}".format(t1[d1]),dates[d1].strftime("(%d-%m)"),"&",
                        "{:6.3f}".format(t2[d2]),dates[d2].strftime("(%d-%m)"),"&",
                        "{:6.3f}".format(t3[d3]),dates[d3].strftime("(%d-%m)"),"\\\ \hline  ")
  print("\end{tabular}")
  print(caption)
  print("\label{table:case"+case+"}")
  print("\end{table}")





r1 = np.load("result_Ny00800_Nt00800_1.npy")
r2 = np.load("result_Ny00800_Nt00800_2.npy")
r3 = np.load("result_Ny00800_Nt00800_3.npy")
r4 = np.load("result_Ny00100_Nt00100_4.npy")

#table1(r1,0,"\caption{Outbreak of a new disease (case I): the maximum expected information gain of each measurement is shown, for every canton. The corresponding date is shown in parenthesis as well. Note that the expected information gain per measurement is defined in \\ref{eq:info_gain}.}","1")
#table1(r2,21,"\caption{Monitoring of social distancing measures (case II): the maximum expected information gain of each measurement is shown, for every canton. The corresponding date is shown in parenthesis as well. Note that the expected information gain per measurement is defined in \\ref{eq:info_gain}.}","2")
#table1(r3,102,"\caption{Loosening of measures, uninformed infection rate (case III): the maximum expected information gain of each measurement is shown, for every canton. The corresponding date is shown in parenthesis as well. Note that the expected information gain per measurement is defined in \\ref{eq:info_gain}.}","3a")
table1(r4,136,"\caption{Loosening of measures, informed infection rate (case III): the maximum expected information gain of each measurement is shown, for every canton. The corresponding date is shown in parenthesis as well. Note that the expected information gain per measurement is defined in \\ref{eq:info_gain}.}","3b")
