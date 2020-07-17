#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import sys
from osp import *
from mpl_toolkits.mplot3d import axes3d
from matplotlib.pyplot import figure
import json
import itertools
from matplotlib import animation
import matplotlib.colors
import matplotlib.patches as patches
import collections

name = ['AG','AI','AR','BE','BL','BS','FR','GE','GL','GR',\
        'JU','LU','NE','NW','OW','SG','SH','SO','SZ','TG',\
        'TI','UR','VD','VS','ZG','ZH']

CANTON_TO_INDEX = {'AG': 0 , 'AI': 1 , 'AR': 2 , 'BE': 3 , 'BL': 4 , 'BS': 5 ,\
                   'FR': 6 , 'GE': 7 , 'GL': 8 , 'GR': 9 , 'JU': 10, 'LU': 11,\
                   'NE': 12, 'NW': 13, 'OW': 14, 'SG': 15, 'SH': 16, 'SO': 17,\
                   'SZ': 18, 'TG': 19, 'TI': 20, 'UR': 21, 'VD': 22, 'VS': 23,\
                   'ZG': 24, 'ZH': 25}

NUM_CANTONS = len(CANTON_TO_INDEX)


def hide_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)

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

code_to_center_shift = {
    'BE':(0,0),
    'LU':(0,0),
    'UR':(0,0),
    'SZ':(0,0),
    'OW':(0,-3),
    'NW':(2,2),
    'GL':(0,0),
    'ZG':(-2,-2),
    'FR':(5,-3),
    'SO':(-5,0),
    'BS':(-1,0.5),
    'BL':(9,-4),
    'SH':(-4,1),
    'AR':(-9,-6),
    'AI':(3,-3),
    'SG':(-7,-3),
    'GR':(0,0),
    'AG':(4,5),
    'TG':(5,7),
    'TI':(0,5),
    'VD':(-20,5),
    'VS':(0,0),
    'NE':(1,-3),
    'GE':(0,-2),
    'JU':(0,3),
        }


for key in code_to_center_shift:
    shift = code_to_center_shift[key]
    k = 1e3
    code_to_center_shift[key] = [shift[0] * k, shift[1] * k]


name_to_code = {}
for code,name in code_to_name.items():
    name_to_code[name] = code

codes = code_to_name.keys()

class Renderer:
    def __init__(self, frame_callback):
        '''
        frame_callback: callable
            Function that takes Renderer and called before rendering a frame.
            It can use `set_values()` and `set_texts()` to update the state,
            and `get_frame()` and `get_max_frame()` to get current frame index.
        '''

        self.frame_callback = frame_callback
        self.code_to_value = {}
        self.code_to_text = {}
        self.code_to_text2 = {}
        self.code_to_bar = {}
        self.codes = codes

        d = np.load('canton_shapes.npy', allow_pickle=True).item()

        # Compute shape centers.
        centers = {}
        self.centers = centers
        for name, ss in d.items():
            for i,s in enumerate(ss):
                x, y = s
                code = name_to_code[name]
                centers[code] = [x.mean(), y.mean()]
                if code in code_to_center_shift:
                    shift = code_to_center_shift[code]
                    centers[code][0] += shift[0]
                    centers[code][1] += shift[1]
                break

        self.set_base_colors('red')
        #self.set_base_colors('green')

        dpi = 200
        fig, ax = plt.subplots(figsize=(1920 / dpi, 1080 / dpi), dpi=dpi)
        hide_axis(ax)
        ax.set_aspect('equal')
        fig.tight_layout()
        self.fig = fig
        self.ax = ax

        # Draw shapes.
        fills = collections.defaultdict(list)
        self.fills = fills
        for name, ss in d.items():
            code = name_to_code[name]
            for i,s in enumerate(ss):
                x, y = s
                line, = ax.plot(x, y, marker=None, c='black', lw=0.25)
                fill, = ax.fill(x, y, alpha=0.25, c='white')
                fills[code].append(fill)

        # Draw labels.
        texts = dict()
        self.texts = texts
        rectacles = dict()
        self.rectacles = rectacles
        for code in codes:
            xc, yc = centers[code]
            ax.text(xc, yc, code, ha='center', va='bottom', zorder=10,color=[0,0,0])
            text = ax.text(
                    xc, yc - 1700,
                    '', ha='center', va='top', zorder=10, fontsize=7,
                    color=[0,0,0])
            texts[code] = text
            ax.scatter(xc, yc, color='black', s=8, zorder=5)
            if code == "SO":
                rectacles[code] = patches.Rectangle((xc+5000-17000,yc-15000), width = 5000, height=1,zorder = 100)
            elif code == "OW":
                rectacles[code] = patches.Rectangle((xc+5000-14000,yc-2000), width = 5000, height=1,zorder = 100)
            elif code == "AR":
                rectacles[code] = patches.Rectangle((xc+5000,yc+8000), width = 5000, height=1,zorder = 100)
            else:
                rectacles[code] = patches.Rectangle((xc+5000,yc), width = 5000, height=1,zorder = 100)
            ax.add_patch(rectacles[code])


        texts2 = dict()
        self.texts2 = texts2
        for code in codes:
            xc, yc = centers[code]
            ax.text(xc, yc, code, ha='center', va='bottom', zorder=10,color=[0,0,0])
            text = ax.text(xc, yc - 1700,'', ha='center', va='top', zorder=10, fontsize=7,color=[0,0,0])
            texts2[code] = text



    def set_values(self, code_to_value):
        '''
        code_to_value: `dict`
          Mapping from canton code to float between 0 and 1.
        '''
        self.code_to_value = code_to_value

    def set_bars(self, code_to_bar):
        '''
        code_to_bar: `dict`
          Mapping from canton code to float between 0 and 1.
        '''
        self.code_to_bar = code_to_bar

    def set_texts(self, code_to_text):
        '''
        code_to_text: `dict`
          Mapping from canton code to label text.
        '''
        self.code_to_text = code_to_text

    def get_values(self):
        return self.code_to_value

    def get_bars(self):
        return self.code_to_bar

    def get_texts(self):
        return self.code_to_text

    def get_codes(self):
        return self.codes

    def get_frame(self):
        '''
        Returns current frame index between 0 and get_max_frame().
        '''
        return self.frame

    def get_max_frame(self):
        '''
        Returns maximum frame index.
        Set to `frames - 1` by `save_movie(frames)`.
        Set to 1 by `save_image()`.
        '''
        return self.max_frame

    def init(self):
        return [v for vv in self.fills.values() for v in vv] + list(self.texts.values())

    def update(self, frame=-1, silent=False):
        self.frame = frame
        self.frame_callback(self)

        if frame == -1:
            frame = self.max_frame
        if not silent:
            print("{:}/{:}".format(frame, self.max_frame))
        for code,value in self.code_to_value.items():
            color = self.base_colors[code]
            alpha = np.clip(value, 0, 1)
            for fill in self.fills[code]:
                fill.set_color(color)
                fill.set_alpha(alpha)


        for code,value in self.code_to_bar.items():
            xc, yc = self.centers[code]
            xc += 5000
            h = int(value*10000)
            if code == "SO":
               xc -= 17000
               yc -= 15000
            if code == "OW":
               xc -= 14000
               yc -= 2000
            if code == "AR":
               xc -= 0
               yc -= -8000
#            p = patches.Rectangle((xc,yc), width = 5000, height=h,zorder = 100)
#            self.ax.add_patch(p)

            self.rectacles[code].set_height(h)

            val = str("{:.3f}".format(value))
            self.texts2[code].set_text(val)
            self.texts2[code].set_position((xc,yc+h+10000))

 


        for code,text in self.code_to_text.items():
            self.texts[code].set_text(str(text))
        return [v for vv in self.fills.values() for v in vv] + list(self.texts.values())

    def save_movie(self, frames=100, filename="a.mp4", fps=15):
        self.max_frame = frames - 1
        ani = animation.FuncAnimation(self.fig, self.update,
                frames=list(range(frames)),
                init_func=self.init, blit=True)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=2000)
        ani.save(filename, writer=writer)

    def save_image(self, frame=-1, filename="a.png"):
        self.max_frame = 1
        self.init()
        self.update(self.max_frame, silent=True)
        self.fig.savefig(filename)

    def set_base_colors(self, code_to_rgb=None):
        if code_to_rgb is None:
            code_to_rgb = {}
            plt_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            for i,code in enumerate(self.get_codes()):
                code_to_rgb[code] = plt_cycle[i % len(plt_cycle)]
        if isinstance(code_to_rgb, str):
            code_to_rgb = {code: code_to_rgb for code in self.get_codes()}
        self.base_colors = {}
        for code in code_to_rgb:
            self.base_colors[code] = np.array(matplotlib.colors.to_rgb(
                    code_to_rgb[code]))



#################################
def make_movie(result,utility,n):
#################################
    from datetime import datetime, timedelta
    """
    v = utility[number of sensors][canton][day]
    """
    days = utility.shape[2]
    base = datetime(2020, 2, 25)
    dates  = np.array([base + timedelta(hours=(24 * i)) for i in range(days)])

    def frame_callback(rend):
        t = rend.get_frame() * (days - 1) // rend.get_max_frame()
        util = utility[n,:,t]
        res  = result[t,:]
        max_res = np.max(result)
        v_u = {}
        v_r1 = {}
        v_r2 = {}
        texts = {}
        for i, c in enumerate(rend.get_codes()):
            i_state = CANTON_TO_INDEX[c]
            v_u[c] = util[i_state]
            v_r1[c] = res [i_state]/max_res
            v_r2[c] = res [i_state]
            tt = (np.asarray(v_r2[c])).astype(int)
            texts[c] = tt.astype(str)
        rend.set_values(v_r1)
        rend.set_texts(texts)
        rend.set_bars(v_u)
        label = dates[t].strftime("%d %b, %Y")
        plt.suptitle(label, fontsize=12)       
    rend = Renderer(frame_callback)
    rend.save_movie(frames=days,filename=str(n) + ".mp4")


##########################
if __name__ == '__main__':
##########################  
  parser = argparse.ArgumentParser()

  parser.add_argument('--result',help='utility result.npy file',type=str)
  parser.add_argument('--output',help='model evaluations output.npy file',type=str)

  args = vars(parser.parse_args())

  print(args["result"])
  print(args["output"])

  utility = np.load(args["result"])
  
  results = np.load(args["output"])
  print (results)
  make_movie(results,utility,0)
  #for n in range(0,utility.shape[0]):
  #    make_movie(res,utility,n)