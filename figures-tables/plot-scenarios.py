import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 5})

import datetime
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from matplotlib.dates import DateFormatter


data      = np.load("canton_daily_cases.npy")
cantons   = data.shape[0] # = 26
days_data = data.shape[1]
name = ['AG','AI','AR','BE','BL','BS','FR','GE','GL','GR',\
        'JU','LU','NE','NW','OW','SG','SH','SO','SZ','TG',\
        'TI','UR','VD','VS','ZG','ZH']

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def prepareData(days = -1,country = False):    
    data = np.load("canton_daily_cases.npy")
    cantons = data.shape[0] # = 26
    if days == -1:
        days = data.shape[1]
    threshold = 0.0
    print("DAYS=",days)
    print("country=",country)
    y = []
    if country == True:
        for d in range(days):
            tot = 0.0
            for c in range(cantons):
                if np.isnan(data[c,d]) == False:
                    tot += data[c,d]
            y.append(tot) 
        return y
    for c in range(cantons):
        d_ = np.copy(data[c,:])
        nans, x= nan_helper(d_)
        d_[nans]= np.interp(x(nans), x(~nans), d_[~nans])
        if np.max(d_) < threshold :
            continue
        print ("cantons:" , c)
        d1 = np.copy(data[c,:])
        for d in range(days):
            if np.isnan(d1[d]) == False:
                y.append(c)
                y.append(d)
                y.append(d1[d])
    return y

import matplotlib.colors as mcolors
COLORS = mcolors.CSS4_COLORS

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
    locator2 = mdates.WeekdayLocator(interval=1)
    formatter = mdates.ConciseDateFormatter(locator)
    date_form = DateFormatter("%b %d")
    ax.xaxis.set_major_locator(locator2)    
    ax.xaxis.set_minor_locator(locator)    
    ax.xaxis.set_major_formatter(date_form)


    ax.axvspan(dates[0 ], dates[21], alpha=0.4, color=COLORS['lightskyblue'])
    ax.axvspan(dates[21], dates[102], alpha=0.4, color=COLORS['lightsalmon'])
    ax.axvspan(dates[102], dates[-1], alpha=0.4, color=COLORS['lightgreen'])

    ax.bar(dates[:102],reference[:102], width=1,label='Daily reported cases',color=COLORS["dimgrey"],zorder=10)
    ax.set_ylabel("Daily Reported infectious")
    ax.set_xlabel("Days ($t$)")
    # fig.legend()
    for label in ax.get_xticklabels():
        label.set_rotation(40)
        label.set_horizontalalignment('right')
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    fig.set_size_inches(3.42, 1.5)
    # plt.tight_layout()
    # plt.show()
    fig.savefig("scenarios.pdf",dpi=100 ,format="pdf")

if __name__ == '__main__':
    plot_scenarios()
