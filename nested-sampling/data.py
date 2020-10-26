#!/usr/bin/env python3
import numpy as np
import os,sys
sys.path.append('../../covid19/')
import epidemics.data.swiss_cantons as swiss_cantons
#import epidemics.data.cases as cases

def get_canton_reference_data():
    cases_per_country = swiss_cantons.fetch_openzh_covid_data()
    return cases_per_country

CANTON_TO_INDEX = {'AG': 0 , 'AI': 1 , 'AR': 2 , 'BE': 3 , 'BL': 4 ,\
                   'BS': 5 , 'FR': 6 , 'GE': 7 , 'GL': 8 , 'GR': 9 ,\
                   'JU': 10, 'LU': 11, 'NE': 12, 'NW': 13, 'OW': 14,\
                   'SG': 15, 'SH': 16, 'SO': 17, 'SZ': 18, 'TG': 19,\
                   'TI': 20, 'UR': 21, 'VD': 22, 'VS': 23, 'ZG': 24,\
                   'ZH': 25}
name = ['AG','AI','AR','BE','BL','BS','FR','GE','GL','GR',\
        'JU','LU','NE','NW','OW','SG','SH','SO','SZ','TG',\
        'TI','UR','VD','VS','ZG','ZH']

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


def prepareData(days = -1,country = False):    
    cantons = 26
    '''
    IR = get_canton_reference_data()
    days = len(IR['TI'])
    data = np.zeros((cantons,days))
    for c in range(cantons):
        c_i = name[c]
        data[c,0] = IR[c_i][0]
        for d in range(1,days):
            data[c,d] = IR[c_i][d] - IR[c_i][d-1]
    np.save("canton_daily_cases.npy",data)
    '''
    data = np.load("canton_daily_cases.npy")

    if days == -1:
    	days = data.shape[1]
    
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
        d1 = np.copy(data[c,:])
        for d in range(days):
            if np.isnan(d1[d]) == False:
                y.append(c)
                y.append(d)
                y.append(d1[d])
    return y
