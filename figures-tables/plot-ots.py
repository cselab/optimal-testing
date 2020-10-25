import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import datetime
import os

plt.rcParams.update({'font.size': 5
    })

CANTON_NAMES = np.array(['AG','AI','AR','BE','BL','BS','FR','GE','GL','GR',\
        'JU','LU','NE','NW','OW','SG','SH','SO','SZ','TG',\
        'TI','UR','VD','VS','ZG','ZH'])

CANTON_TO_INDEX = {'AG': 0 , 'AI': 1 , 'AR': 2 , 'BE': 3 , 'BL': 4 , 'BS': 5 ,\
                   'FR': 6 , 'GE': 7 , 'GL': 8 , 'GR': 9 , 'JU': 10, 'LU': 11,\
                   'NE': 12, 'NW': 13, 'OW': 14, 'SG': 15, 'SH': 16, 'SO': 17,\
                   'SZ': 18, 'TG': 19, 'TI': 20, 'UR': 21, 'VD': 22, 'VS': 23,\
                   'ZG': 24, 'ZH': 25}

COLORS = mcolors.CSS4_COLORS
COLOR_NAMES = ["royalblue", "forestgreen", "gold", "tomato"]
COLOR_NAMES_SHADES = [["darkblue", "darkolivegreen", "gold", "darkred"], ["blue", "darkgreen", "yellow", "red"], ["royalblue", "forestgreen", "lightyellow", "tomato"]]
LINECOLOR_NAMES = ["blue", "darkgreen", "orange", "darkred"]

code_to_name = {
    'ZH':'Zürich',
    'BE':'Bern',
    'LU':'Luzern',
    'UR':'Uri',
    'SZ':'Schwyz',
    'OW':'Obwalden',
    'NW':'Nidwalden',
    'GL':'Glarus',
    'ZG':'Zug',
    'FR':'Fribourg',
    'SO':'Solothurn',
    'BS':'Basel-Stadt',
    'BL':'Basel-Landschaft',
    'SH':'Schaffhausen',
    'AR':'Appenzell Ausserrhoden',
    'AI':'Appenzell Innerrhoden',
    'SG':'St. Gallen',
    'GR':'Graubünden',
    'AG':'Aargau',
    'TG':'Thurgau',
    'TI':'Ticino',
    'VD':'Vaud',
    'VS':'Valais',
    'NE':'Neuchâtel',
    'GE':'Genève',
    'JU':'Jura',
}

name_to_code = {}
for code,name in code_to_name.items():
    name_to_code[name] = code

codes = code_to_name.keys()

code_to_center_shift = {
    'BE':(0,0),
    'LU':(0,0),
    'UR':(0,0),
    'SZ':(0,0),
    'OW':(0,-5),
    'NW':(2,0),
    'GL':(0,0),
    'ZG':(-2,-5),
    'FR':(5,-3),
    'SO':(-5,0),
    'BS':(-1,0.5),
    'BL':(9,-4),
    'SH':(-4,-1),
    'AR':(-9,-7),
    'AI':(2,-6),
    'SG':(-7,-3),
    'GR':(0,0),
    'AG':(4,3),
    'TG':(5,5),
    'TI':(0,5),
    'VD':(-20,5),
    'VS':(0,0),
    'NE':(1,-3),
    'GE':(0,-6),
    'JU':(0,3),
    'ZH':(0,-2)
        }

def getQuantiles( data, disp ):
    samples = []
    for d in range(data.shape[0]):
        sample = []
        for i in range(data.shape[1]):
            total_cases = np.zeros(1000)
            for c in range(data.shape[3]):
                mean        = data[d,i,i,c]
                dispersion  = disp[i] * mean +  1e-10
                pr          = 1.0 / (1.0 + mean/dispersion)
                total_cases += np.random.negative_binomial(n=dispersion, p=pr, size=1000)
            sample += total_cases.tolist()
        samples.append(sample)

    p = 0.99
    q50  = np.quantile ( a= samples , q = 0.50  , axis = 1)
    qlo  = np.quantile ( a= samples , q = 0.5 - p/2 , axis = 1)
    qhi  = np.quantile ( a= samples , q = 0.5 + p/2 , axis = 1)
    return [qlo, qhi]

class utility:
    def __init__(self, filename):
        ## data
        self.data = np.load(filename)
        self.dims = self.data.shape
        print(self.dims)
        self.nSensors = 4
        self.nCantons = self.dims[1]
        self.nTimesteps = self.dims[2]

        self.maxTime_Canton = self.data.max(2)
        self.argmaxTime = self.data.argmax(2)

        self.maxCanton_Time = self.data.max(1)
        self.argmaxSpace = self.data.argmax(1)

        self.max = self.maxTime_Canton.max(1)
        self.maxCanton = self.maxTime_Canton.argmax(1)
        self.maxTime = self.maxCanton_Time.argmax(1)

        ## dates
        self.base    = datetime.datetime(2020, 2, 25) #February 25th, 2020
        self.dates   = np.array([self.base + datetime.timedelta(hours=(24 * i)) for i in range(self.nTimesteps)])
        self.locator = mdates.DayLocator(interval=1)
        self.locator2 = mdates.WeekdayLocator(interval=2)
        formatter = mdates.ConciseDateFormatter(self.locator)
        self.date_form = DateFormatter("%b %d")


    def plotUtilities(self):
        fig, ax = plt.subplots(6,5)
        permutation = [ -1, -4, 4, 3 ] + [i for i in range(4,22)] + [ 1,23,24,0]
        # plot utility plots
        for ik in range(6):
            for jk in range(5):
                idx = ik*5+jk
                if idx>= self.nCantons:
                    fig.delaxes(ax[ik][jk])
                else:
                    ax[ik][jk].set_ylim([0,6])
                    ax[ik][jk].set_ylabel("$\hat U(t,{})$".format(CANTON_NAMES[permutation[idx]]))
                    ax[ik][jk].set_xlabel("Days ($t$)")
                    ax[ik][jk].set_title("Canton {}".format(CANTON_NAMES[permutation[idx]]))
                    ax[ik][jk].xaxis.set_major_locator(self.locator)  
                    ax[ik][jk].xaxis.set_major_formatter(self.date_form)
                    for s in range(self.nSensors):
                        if s == 0:
                            ax[ik][jk].fill_between(self.dates, 0,self.data[s][permutation[idx]],color=COLORS[COLOR_NAMES[s]])
                        else:
                            ax[ik][jk].fill_between(self.dates, self.max[s-1],self.data[s][permutation[idx]],color=COLORS[COLOR_NAMES[s]])
                            
                        ax[ik][jk].plot(self.dates, self.data[s][permutation[idx]],color=COLORS[LINECOLOR_NAMES[s]])
                        for label in ax[ik][jk].get_xticklabels():
                            label.set_rotation(40)
                            label.set_horizontalalignment('right')
        # indicate optimal location
        for s in range(self.nSensors):
            ik = int(self.maxCanton[s]//5)
            jk = int(self.maxCanton[s]%5)
            ax[0][s].plot(self.dates[self.maxTime[s]],self.max[s]+0.3,marker="${}$".format(s+1),zorder=10,color="black", markersize=3)
            ax[0][s].vlines(x=self.dates[self.maxTime[s]],ymin=0, ymax=self.max[s], color="black", linestyle="dashed", linewidth=0.75)
        fig.set_size_inches(7, 8.75)
        fig.tight_layout()
        plt.savefig("Utility.eps", format='eps')

    def plotMaxUtility(self):
        shapes = np.load('../canton_shapes.npy', allow_pickle=True).item()
        k = np.arange(self.nCantons)
        fig, ax = plt.subplots(4,2,sharey="row")
        ax[1,0].set_ylabel("Expected Information Gain")
        ax[3,0].set_ylabel("Expected Information Gain")
        for sens in range(4):
            ax[sens//2*2,sens%2].axis('off')
            ax[sens//2*2,sens%2].set_aspect('equal')
            for name, coords in shapes.items():
                code = name_to_code[name]
                # plot cantons
                for _,coord in enumerate(coords):
                    x, y = coord
                    ax[sens//2*2,sens%2].plot(x, y, marker=None, c='black', lw=0.25)
                    fill, = ax[sens//2*2,sens%2].fill(x, y, alpha=0.25, c='white')
                    fill.set_color(COLORS[COLOR_NAMES[sens]])
                    fill.set_alpha(np.exp(self.maxTime_Canton[sens][CANTON_TO_INDEX[code]])/np.exp(self.max[sens]))

            # Compute shape centers and plot canton label
            centers = {}
            for name, ss in shapes.items():
                for i,s in enumerate(ss):
                    x, y = s
                    code = name_to_code[name]
                    centers[code] = [x.mean(), y.mean()]
                    if code in code_to_center_shift:
                        shift = code_to_center_shift[code]
                        centers[code][0] += 1e3*shift[0]
                        centers[code][1] += 1e3*shift[1]
                    break

            for code in codes:
                xc, yc = centers[code]
                ax[sens//2*2,sens%2].text(xc, yc, code, ha='center', va='bottom', zorder=10,color=[0,0,0])

            idxSort = np.argsort(-self.maxTime_Canton[sens])
            if sens == 0:
                ax[sens//2*2+1,sens%2].bar(k, self.maxTime_Canton[sens][idxSort], tick_label=CANTON_NAMES[idxSort], width=0.8, color=COLORS[COLOR_NAMES[sens]])
            else:
                ax[sens//2*2+1,sens%2].bar(k, self.maxTime_Canton[sens][idxSort]-self.max[sens-1], tick_label=CANTON_NAMES[idxSort], width=0.8, color=COLORS[COLOR_NAMES[sens]])

            for c in range(self.nCantons):
                opt_date = "{:02}-{:02}".format(self.dates[self.argmaxTime[sens][c]].month,self.dates[self.argmaxTime[sens][c]].day)
                if sens == 0:
                    ax[sens//2*2+1,sens%2].text(k[c], self.maxTime_Canton[sens][idxSort][c]+0.2, opt_date, ha='center', va='bottom', zorder=10,rotation=90)
                else:
                    ax[sens//2*2+1,sens%2].text(k[c], self.maxTime_Canton[sens][idxSort][c]-self.max[sens-1]+0.2, opt_date, ha='center', va='bottom', zorder=10,rotation=90)
            ax[sens//2*2+1,sens%2].set_ylim([0,3.5])
            ax[sens//2*2+1,sens%2].set_xticklabels(ax[sens//2*2+1,sens%2].get_xticklabels(), rotation=45)
            ax[sens//2*2+1,sens%2].spines['right'].set_visible(False)
            ax[sens//2*2+1,sens%2].spines['top'].set_visible(False)
            ax[sens//2*2+1,sens%2].set_xlabel("Canton ($k$)")
            ax[sens//2*2+1,sens%2].set_title('Additional Expected Information Gain for Survey {}'.format(sens+1))

        fig.set_size_inches(7, 9)
        fig.tight_layout()
        plt.savefig("MaxUtilities.pdf")

    def plotSecondOutbreak(self):
        ## PLOT CONFIDENCE INTERVALS ##
        samples_per_day = 1
        data = np.load("runs.npy")
        disp = np.load("dispersion.npy")
        
        data1 = np.load("runs2.npy")
        disp1 = np.load("dispersion2.npy")

        numDays = data1.shape[0]
        dates   = np.array([self.base + datetime.timedelta(hours=(24 * i)) for i in range(numDays)])

        quantiles = getQuantiles( data, disp )
        quantiles1 = getQuantiles( data1, disp1 )
 
        fig, ax = plt.subplots(2,1,sharex=True)
        ax[0].fill_between(dates[:quantiles[0].shape[0]], quantiles[0], quantiles[1],color=COLORS["lightgrey"])
        ax[1].fill_between(dates, quantiles1[0], quantiles1[1],color=COLORS["lightgrey"])

        ## PLOT UTILITIES 3a ##
        ax2 = ax[0].twinx()
        k = np.arange(self.nCantons)
        for sens in range(4):
            idxSort = np.argsort(-self.maxTime_Canton[sens])
            for c in range(1):
                col = COLORS[COLOR_NAMES_SHADES[c][sens]]
                # FOR SENS==0 START WITH 0
                if sens == 0:
                    ax2.bar(dates[self.argmaxTime[sens][idxSort]][c], self.maxTime_Canton[sens][idxSort][c], width=1.5, color=col,zorder=2+c)
                    ax2.text(dates[self.argmaxTime[sens][idxSort]-1][c],(self.maxTime_Canton[sens][idxSort][c]), CANTON_NAMES[idxSort][c],ha='right', va='center')
                # TAKE DIFF OTHERWISE
                else:
                    ax2.bar(dates[self.argmaxTime[sens][idxSort]][c], (self.maxTime_Canton[sens][idxSort][c]-self.max[sens-1]), width=1.5, color=col,zorder=3+sens+c)
                    ax2.text(dates[self.argmaxTime[sens][idxSort]-1][c],(self.maxTime_Canton[sens][idxSort][c]-self.max[sens-1]), CANTON_NAMES[idxSort][c],ha='right', va='center', zorder=10)

        ## PLOT UTILITIES 3b ##
        ## data
        data = np.load("result_Ny00800_Nt00800_2.npy")
        dims = data.shape

        nSensors = 4
        nCantons = dims[1]
        nTimesteps = dims[2]

        maxTime_Canton = data.max(2)
        argmaxTime = data.argmax(2)

        max = maxTime_Canton.max(1)

        ax2b = ax[1].twinx()
        for sens in range(4):
            idxSort = np.argsort(-maxTime_Canton[sens])
            for c in range(1):
                col = COLORS[COLOR_NAMES_SHADES[c][sens]]
                # FOR SENS==0 START WITH 0
                if sens == 0:
                    ax2b.bar(dates[argmaxTime[sens][idxSort]][c], maxTime_Canton[sens][idxSort][c], width=1.5, color=col,zorder=2+c)
                    ax2b.text(dates[argmaxTime[sens][idxSort]-1][c],(maxTime_Canton[sens][idxSort][c]), CANTON_NAMES[idxSort][c],ha='right', va='center')
                # TAKE DIFF OTHERWISE
                else:
                    ax2b.bar(dates[argmaxTime[sens][idxSort]][c], (maxTime_Canton[sens][idxSort][c]-max[sens-1]), width=1.5, color=col,zorder=3+sens+c)
                    ax2b.text(dates[argmaxTime[sens][idxSort]-1][c],(maxTime_Canton[sens][idxSort][c]-max[sens-1]), CANTON_NAMES[idxSort][c],ha='right', va='center', zorder=10)

        ## PLOT DATA ##
        data = np.load("../canton_daily_cases.npy")
        cantons = data.shape[0] 
        days = data.shape[1]
        dates   = np.array([self.base + datetime.timedelta(hours=(24 * i)) for i in range(days)])
        y = []

        for d in range(days):
            tot = 0.0
            for c in range(cantons):
                if np.isnan(data[c,d]) == False:
                    tot += data[c,d]
            y.append(tot)
        ax[0].plot(dates[:102],y[:102],'.',label='data',color="black",zorder=10, markersize=1)
        ax[1].plot(dates[:136],y[:136],'.',label='data',color="black",zorder=10, markersize=1)

        ax[0].xaxis.set_major_locator(self.locator2)    
        ax[0].xaxis.set_minor_locator(self.locator)
        ax[1].xaxis.set_major_locator(self.locator2)    
        ax[1].xaxis.set_minor_locator(self.locator)    
        ax[1].xaxis.set_major_formatter(self.date_form)
        ax[0].set_ylabel("Daily Reported Infectious")
        ax[1].set_ylabel("Daily Reported Infectious")
        ax2.set_ylabel("Expected Information Gain")
        ax2b.set_ylabel("Expected Information Gain")
        ax[0].set_ylim([0,1600])
        ax[1].set_ylim([0,1600])
        ax2.set_ylim([0,2.5])
        ax2b.set_ylim([0,2.5])
        for label in ax[1].get_xticklabels():
            label.set_rotation(40)
            label.set_horizontalalignment('right')
        ax[1].set_xlabel("Days ($t$)")
        fig.set_size_inches(3.42, 2.7)
        plt.tight_layout()
        fig.savefig("second-outbreak.eps", format='eps')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--case',help='integer indicating which case to plot',type=int, required=True)
  args = vars(parser.parse_args())

  os.chdir("./case{}".format(args["case"]))
  util = utility("result_Ny00800_Nt00800.npy")

  if args["case"] == 1:
    util.plotUtilities()
  if args["case"] == 2:
    util.plotMaxUtility()
  if args["case"] == 3:
    util.plotSecondOutbreak()