import numpy as np

def prepareData(days = -1,country = False):    

    data = np.load("canton_daily_cases.npy")
    cantons = data.shape[0] # = 26
    if days == -1:
    	days = data.shape[1]

    y = []
    if country == True:
        for d in range(days):
            tot = 0.0
            for c in range(cantons):
                if np.isnan(data[c,d]) == False:
                    tot += data[c,d]
            y.append(tot) 
    else:
      for c in range(cantons):
          d_ = data[c,:]
          for d in range(days):
              if np.isnan(d_[d]) == False:
                  y.append(c)
                  y.append(d)
                  y.append(d_[d])
    return y
