#!/usr/bin/env python3

CANTONS = 26
NAMES = ['AG','AI','AR','BE','BL','BS','FR','GE','GL','GR',\
         'JU','LU','NE','NW','OW','SG','SH','SO','SZ','TG',\
         'TI','UR','VD','VS','ZG','ZH']
'''
Below we list the dates that interest us for this particular study:
Day 0   (25.02.2020) : First reported case in Ticino
Day 21  (17.03.2020) : First measures announced by the government
Day 102 (06.06.2020) : Loosening of measures started
'''

#Case I: Start of the epidemic
#No data is available and sensors are placed in the first week.
T_DATA_CASE_1 = 0 
T_S_CASE_1 = 0
T_E_CASE_1 = 8

#Case II: First intervention announced
#Data is available for the first 21 days and sensors are placed in the next two weeks.
T_DATA_CASE_2 = 21
T_S_CASE_2 = 21
T_E_CASE_2 = 21 + 14

#Case III: Loosening of measures annoucned
#Data is available for the first 102 days and sensors are placed in the next 38 days.
T_DATA_CASE_3 = 21
T_S_CASE_3 = 102
T_E_CASE_3 = 102 + 38
