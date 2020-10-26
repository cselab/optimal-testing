import pickle
import sys
import os

def printerr(m):
    sys.stderr.write(str(m) + '\n')

def get_cache_name(name):
    return "{:}.cache".format(name)

def load(name):
    c = get_cache_name(name)
    if os.path.isfile(c):
        printerr("cache hit: {:}".format(c))
        with open(c, 'rb') as f:
            return pickle.load(f)
    printerr("cache miss: {:}".format(c))
    return None

def save(name, r):
    c = get_cache_name(name)
    printerr("save to cache: {:}".format(c))
    with open(c, 'wb') as f:
        pickle.dump(r, f)
    return r
