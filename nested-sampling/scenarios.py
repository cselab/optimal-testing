import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

from data import *
import datetime
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
from matplotlib.dates import DateFormatter

data      = np.load("canton_daily_cases.npy")
cantons   = data.shape[0] # = 26
days_data = data.shape[1]
name = ['AG','AI','AR','BE','BL','BS','FR','GE','GL','GR',\
        'JU','LU','NE','NW','OW','SG','SH','SO','SZ','TG',\
        'TI','UR','VD','VS','ZG','ZH']

def plot_scenarios():
    days       = days_data 

    fig, ax = plt.subplots(constrained_layout=True)
    reference = prepareData(country = True)
    
    prediction = []
    base    = datetime.datetime(2020, 2, 25) #February 25th, 2020
    dates   = np.array ([base + datetime.timedelta(hours=(24 * i)) for i in range(days)])
    dates2   = np.array([base + datetime.timedelta(hours=(24 * i)) for i in range(days+30)])

    locator   = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)    

    locator = mdates.DayLocator(interval=1)
    locator2 = mdates.WeekdayLocator(interval=2)
    formatter = mdates.ConciseDateFormatter(locator)
    date_form = DateFormatter("%b %d")
    ax.xaxis.set_major_locator(locator2)    
    ax.xaxis.set_minor_locator(locator)    
    ax.xaxis.set_major_formatter(date_form)


    ax.axvspan(dates[0 ], dates[21], alpha=0.4, color='red')
    ax.axvspan(dates[21], dates[-9], alpha=0.4, color='green')
    ax.axvspan(dates2[-39], dates2[-1], alpha=0.4, color='blue')

    ax.bar(dates,reference, width=0.6,label='Daily reported cases',color="black")
    fig.legend()
    plt.show()

    fig.savefig("scenarios.pdf",dpi=100 ,format="pdf")

plot_scenarios()
