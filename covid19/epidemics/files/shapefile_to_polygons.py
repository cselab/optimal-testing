#!/usr/bin/env python3

from mpl_toolkits.basemap import Basemap
import numpy as np
import sys
import os
import shapefile

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from epidemics.data.swiss_cantons import get_shape_file

sh = shapefile.Reader(get_shape_file())
d = {}
for r in sh.shapeRecords():
    rd = r.record.as_dict()
    name = rd['NAME']
    coords = r.shape.__geo_interface__['coordinates']
    xy = np.array(coords[0]).T.astype(np.float32)
    if name not in d:
        d[name] = []
    d[name].append(xy)

np.save("canton_shapes.npy", d)
