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

    Arguments:
        region_keys: List of region names.
        region_population: List of population size of corresponding regions.
        Mij: A numpy matrix of region-region number of commuters.
    """
    def __init__(self, region_keys, region_population, Mij):
        K = len(region_keys)
        assert len(region_population) == K
        assert Mij.shape == (K, K)
        self.num_regions = K
        self.region_keys = region_keys
        self.region_population = region_population
        self.Mij = Mij
        self.key_to_index = {key: k for k, key in enumerate(region_keys)}
    def to_cpp(self):
        """Return the libepidemics.ModelData instance.
        Needed when running the model from Python using the C++ implementation."""
        return libepidemics.ModelData(
                self.region_keys, self.region_population,flatten(self.Mij))


def get_canton_model_data():
    """Creates the ModelData instance with default data."""
    keys = swiss_cantons.CANTON_KEYS_ALPHABETICAL
    population = [swiss_cantons.CANTON_POPULATION[c] for c in keys]
    Mij = swiss_cantons.get_Mij_numpy(keys)

    return ModelData(keys, population, Mij)