#!/usr/bin/env python3

import json
import numpy as np
import sys
import argparse

parser = argparse.ArgumentParser(description=
        "Prints asymmetry |M - M.T| / |M| of a matrix M.")
parser.add_argument('matrix', type=str,
        help="Path to matrix in json format (e.g. M['ZH']['BE']=1000)")
args = parser.parse_args()

with open(args.matrix) as f:
    mj = json.load(f)

cc = mj.keys()

N = len(cc)
m = np.zeros((N,N))

for i0,c0 in enumerate(cc):
    for i1,c1 in enumerate(cc):
        m[i0,i1] += mj[c0][c1]

diff = abs(m - m.T).mean()
mean = abs(m).mean()
print(diff / mean)


