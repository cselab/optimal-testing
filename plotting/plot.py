#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import json
import itertools
from matplotlib import animation
import matplotlib.colors
import matplotlib.patches as patches
import os
import collections

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

if __name__ == "__main__":

    def frame_callback(rend):
        colors = dict()
        texts = dict()
        for i, c in enumerate(rend.get_codes()):
            colors[c] = np.sin(i + rend.get_frame() * 0.1) ** 2
            texts[c] = "{:},{:}".format(rend.get_frame(), i)
        rend.set_values(colors)
        rend.set_texts(texts)

    rend = Renderer(frame_callback)

    rend.save_image()
    rend.save_movie(frames=30)