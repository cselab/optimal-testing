#!/usr/bin/env python3
import numpy as np
import epidemics.data.swiss_cantons as swiss_cantons

name = ['AG','AI','AR','BE','BL','BS',
        'FR','GE','GL','GR','JU','LU',
        'NE','NW','OW','SG','SH','SO',
        'SZ','TG','TI','UR','VD','VS',
        'ZG','ZH']

IR = swiss_cantons.fetch_openzh_covid_data()
days = len(IR['TI'])
cantons = 26
all_data = np.zeros((cantons,days))
for c in range(cantons):
    c_i = name[c]
    all_data[c,0] = IR[c_i][0]
    for d in range(1,days):
        all_data[c,d] = IR[c_i][d] - IR[c_i][d-1]
np.save("canton_daily_cases.npy",all_data)
