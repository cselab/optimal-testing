#!/usr/bin/env python3

import numpy as np

from datetime import datetime
import json
import math
import os
import sys
import urllib.request

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build'))

try:
    import libepidemics
except ModuleNotFoundError:
    sys.exit("libepidemics not found. Did you forget to compile the C++ code?")

from epidemics import DATA_CACHE_DIR
import epidemics.swiss_cantons as swiss_cantons

def flatten(matrix):
    """
    >>> flatten([[10, 20, 30], [40, 50]])
    [10, 20, 30, 40, 50]
    """
    return [
        value
        for row in matrix
        for value in row
    ]



class ModelData:
    """Model data such as region population and Mij matrix.

    For conveniece, we store parameters beta, mu, alpha and other scalars
    separately (as libepidemics.cantons.<model>.Parameters).

    Arguments:
        region_keys: List of region names.
        region_population: List of population size of corresponding regions.
        Mij: A numpy matrix of region-region number of commuters.
        ext_com_Iu: A matrix [day][region] of estimated number of foreign
                    infected people visiting given region at given day.
        Ui: User-defined, shape (K)
    """
    def __init__(self, region_keys, region_population, Mij, Cij, *, Ui=[]):
        K = len(region_keys)
        assert len(region_population) == K
        assert Mij.shape == (K, K)
        assert Cij.shape == (K, K)
        if not len(Ui):
            Ui = [0] * K
        assert len(Ui) == K

        self.num_regions = K
        self.region_keys = region_keys
        self.region_population = region_population
        self.Mij = Mij
        self.Cij = Cij
        #self.ext_com_Iu = ext_com_Iu
        self.Ui = Ui

        self.key_to_index = {key: k for k, key in enumerate(region_keys)}

    def to_cpp(self):
        """Return the libepidemics.ModelData instance.

        Needed when running the model from Python using the C++ implementation."""
        return libepidemics.cantons.ModelData(
                self.region_keys, self.region_population,
                flatten(self.Mij), flatten(self.Cij),
                self.Ui)

    def save_cpp_dat(self, path=DATA_CACHE_DIR / 'cpp_model_data.dat'):
        """Generate cpp_model_data.dat, the data for the C++ ModelData class.

        Needed when running Korali from C++, when `to_cpp` is not available.

        File format:
            <number of regions N>
            abbreviation1 ... abbreviationN
            population1 ... populationN

            M_11 ... M_1N
            ...
            M_N1 ... M_NN

            <number of days D for external cases>
            <external cases for day 1 for region 1> ... <external cases for day 1 for region N>
            ...
            <external cases for day D for region 1> ... <external cases for day D for region N>
        """
        with open(path, 'w') as f:
            f.write(str(self.num_regions) + '\n')
            f.write(' '.join(self.region_keys) + '\n')
            f.write(' '.join(str(p) for p in self.region_population) + '\n\n')

            for row in self.Mij:
                f.write(' '.join(str(x) for x in row) + '\n')
            f.write('\n')

            f.write(str(len(self.ext_com_Iu)) + '\n')
            for day in self.ext_com_Iu:
                f.write(' '.join(str(x) for x in day) + '\n')
            f.write(' '.join(str(u) for u in self.Ui) + '\n')
        print(f"Stored model data to {path}.")



def get_canton_model_data():
    """Creates the ModelData instance with default data."""
    keys = swiss_cantons.CANTON_KEYS_ALPHABETICAL
    population = [swiss_cantons.CANTON_POPULATION[c] for c in keys]

    Mij = swiss_cantons.get_Mij_numpy(keys)
    Cij = swiss_cantons.get_Cij_numpy(keys)

    ext_com_Iu = []  # Data for 0 days.

    return ModelData(keys, population, Mij, Cij) #, ext_com_Iu=ext_com_Iu)
