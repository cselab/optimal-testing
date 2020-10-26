#!/usr/bin/env python3

from matrixconverters.read_ptv import ReadPTVMatrix
import numpy as np
import cache
import geopandas as gpd
import itertools
import collections
import json


def load_matrix(path):
    '''
    path: `str`
        Path to .mtx file.

    Returns:
    matrix: `numpy.ndarray(np.float32)`, (N,N)
        Number of people traveling from zone `i` to zone `j` in `matrix[i,j]`.
    zones: `numpy.ndarray(str)`, (N)
        Name of zone `i` in `zones[i]`.
    '''
    cachename = path
    r = cache.load(cachename)
    if r is not None: return r

    m = ReadPTVMatrix(filename=p)
    matrix = m['matrix'].astype(np.float32)
    ids = [int(z.coords['zone_no'].data) for z in m['zone_name']]

    origins = [int(v.data) for v in matrix['origins']]
    destinations = [int(v.data) for v in matrix['destinations']]
    assert origins == ids, \
            "different order in matrix['origins'] and zone_name"
    assert destinations == ids, \
            "different order in matrix['destinations'] and zone_name"

    zonenames = np.array([str(z.data) for z in m['zone_name']])

    r = matrix.data, zonenames
    return cache.save(cachename, r)

def load_zones(path):
    '''
    path: str
        Path to .gpkg file.

    Returns:
    zone_to_canton: `dict`
        Mapping from zone name to canton code (e.g. 'Dietlikon' -> 'ZH')
    '''
    cachename = path
    r = cache.load(cachename)
    if r is not None: return r

    gdf = gpd.read_file(p)
    zonenames = list(map(str, gdf.N_Gem))
    zonecantons = list(map(str, gdf.N_KT))

    zone_to_canton = {}

    for name,canton in zip(zonenames, zonecantons):
        zone_to_canton[name] = canton

    r = zone_to_canton
    return cache.save(cachename, r)

p = "public_transport.mtx"
matrixzones, zonenames = load_matrix(p)

p = "zones.gpkg"
zone_to_canton = load_zones(p)

cantons = np.unique(list(zone_to_canton.values()))
matrix = collections.defaultdict(dict)

canton_to_idx = {c : i for i,c in enumerate(cantons)}

cc = np.array([canton_to_idx[zone_to_canton[z]] for z in zonenames], dtype=int)

def pair_to_hash(c0, c1):
    return c0 + len(cantons) * c1

matrixzones_cc = pair_to_hash(cc[:,None], cc[None,:])

bins = np.bincount(matrixzones_cc.flatten(), weights=matrixzones.flatten())

for c0,c1 in itertools.product(cantons, repeat=2):
    i0 = canton_to_idx[c0]
    i1 = canton_to_idx[c1]
    matrix[c0][c1] = int(bins[pair_to_hash(i0, i1)])

oname = 'matrix.json'
print(oname)
with open(oname, 'w') as o:
    json.dump(matrix, o)

