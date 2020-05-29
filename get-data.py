import numpy as np

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), './covid19/'))

import epidemics.data.swiss_cantons as swiss_cantons
import epidemics.data.cases as cases

def get_canton_reference_data():
    cases_per_country = swiss_cantons.fetch_openzh_covid_data()
    return cases_per_country

CANTON_TO_INDEX = {'AG': 0, 'AI': 1, 'AR': 2, 'BE': 3, 'BL': 4, 'BS': 5, 'FR': 6, 'GE': 7, 'GL': 8, 'GR': 9, 'JU': 10, 'LU': 11, 'NE': 12, 'NW': 13, 'OW': 14, 'SG': 15, 'SH': 16, 'SO': 17, 'SZ': 18, 'TG': 19, 'TI': 20, 'UR': 21, 'VD': 22, 'VS': 23, 'ZG': 24, 'ZH': 25}

name = ['AG','AI','AR','BE','BL','BS','FR','GE','GL','GR',\
        'JU','LU','NE','NW','OW','SG','SH','SO','SZ','TG',\
        'TI','UR','VD','VS','ZG','ZH']

IR = get_canton_reference_data()
days = len(IR['TI'])
cantons = 26

all_data = np.zeros((cantons,days))
for c in range(cantons):
    c_i = name[c]
    all_data[c,0] = IR[c_i][0]
    for d in range(1,days):
        all_data[c,d] = IR[c_i][d] - IR[c_i][d-1]
np.save("canton_daily_cases.npy",all_data)