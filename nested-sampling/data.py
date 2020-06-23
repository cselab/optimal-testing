import numpy as np

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def prepareData(days = -1,country = False):    

    data = np.load("canton_daily_cases.npy")
    cantons = data.shape[0] # = 26
    if days == -1:
    	days = data.shape[1]
    
    threshold = 50.0
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