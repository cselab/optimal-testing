"""
This file provides access to all relevant data aobut Swiss cantons and COVID-19:
    - list of cantons and their population count
    - connectivity matrix between cantons
    - number of cases per day, for each canton
"""

from epidemics import DATA_CACHE_DIR, DATA_DOWNLOADS_DIR, DATA_FILES_DIR
import numpy as np

import datetime
import os
import pandas as pd

import pathlib
import time
import urllib.request

from pathlib import Path
import functools
import inspect
import json
import pickle


DAY = datetime.timedelta(days=1)



def download(url):
    """Download and return the content of a URL."""
    print(f"[Epidemics] Downloading {url}... ", end="", flush=True)
    req = urllib.request.urlopen(url)
    data = req.read()
    print("Done.", flush=True)
    return data

def cache(func):
    """Caches the result of a given function, depending on the function arguments.

    A decorated function accepts only hashable types, e.g. tuples and not lists.

    Example:
        @cache
        def func(x):
            print(x)
            return x * x

        a = func(10)    # Prints 10.
        b = func(20)    # Prints 20.
        c = func(20)    # Prints nothing (value is cached).
        print(a, b, c)  # 100, 400, 400
    """
    _cache = {}
    def inner(*args, **kwargs):
        key = (args, frozenset(kwargs.items()))
        if key in _cache:
            return _cache[key]
        _cache[key] = out = func(*args, **kwargs)
        return out
    return functools.wraps(func)(inner)


def cache_to_file(target, dependencies=[]):
    """Factory for a decorator that caches the result of a no-argument function and stores it to a target file.

    Handles JSON, pickle and pandas.DataFrame CSV files.

    Arguments:
        target: The target cache filename.
        dependencies: The list of files that the result depends on.

    If the target file exists but is older than any of the dependencies, it will be recomputed.
    """
    target = Path(target)
    dependencies = [Path(d) for d in dependencies]

    target_str = str(target)

    if target_str.endswith('.json'):
        def load(path):
            with open(path) as f:
                return json.load(f)

        def save(content, path):
            with open(path, 'w') as f:
                json.dump(content, f)

    elif target_str.endswith('.pickle'):
        def load(path):
            with open(path, 'rb') as f:
                return pickle.load(f)

        def save(content, path):
            with open(path, 'wb') as f:
                pickle.dump(content, f)

    elif target_str.endswith('.df.csv'):
        load = pd.read_csv

        def save(content, path):
            with open(path, 'w') as f:
                f.write(content.to_csv(index=False))

    else:
        raise ValueError(f"Unrecognized extension '{target.suffix}'. "
                         f"Only .json and .pickle supported.")

    def decorator(func):
        all_dependencies = dependencies + [Path(inspect.getfile(func))]

        def inner():
            try:
                modified_time = target.stat().st_mtime
            except FileNotFoundError:
                pass
            else:
                if all(modified_time > d.stat().st_mtime
                       for d in all_dependencies):
                    print(f"Loading the result of `{func.__name__}` from the cache file `{target}`.")
                    return load(target)

            result = func()
            target.parent.mkdir(parents=True, exist_ok=True)
            save(result, target)
            return result
        return functools.wraps(func)(inner)
    return decorator






def download_and_save(url, path, cache_duration=1000000000, load=True):
    """Download the URL, store to a file, and return its content.

    Arguments:
        url: URL to download.
        path: Target file path.
        cache_duration: (optional) Reload if the file on disk is older than the given duration in seconds.
        load: Should the file be loaded in memory? If not, `None` is returned.
    """
    path = pathlib.Path(path)
    try:
        if time.time() - path.lstat().st_mtime <= cache_duration:
            if load:
                with open(path, 'rb') as f:
                    return f.read()
            elif os.path.exists(path):
                return None
    except FileNotFoundError:
        pass

    data = download(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(data)
    if load:
        return data
    else:
        return None

def extract_zip(zippath, member_pattern, save_dir, overwrite=False):
    """
    Extracts files from zip archive which name contains `member_pattern`,
    saves them to `save_dir`, and returns paths to saved files.
    """
    from zipfile import ZipFile
    from pathlib import Path
    os.makedirs(save_dir, exist_ok=True)
    paths = []
    with ZipFile(zippath, 'r') as zipobj:
        for member in zipobj.namelist():
            if member_pattern in member:
                path = Path(save_dir) / os.path.basename(member)
                if overwrite or not os.path.isfile(path):
                    print("extracting '{:}'".format(member))
                    with open(path, 'wb') as f:
                        f.write(zipobj.read(member))
                paths.append(path)
    return paths


@cache
def bfs_residence_work_xls(header=(3, 4), usecols=None):
    """Return the residence-workplace commute Excel file as a pandas.DataFrame."""
    url = 'https://www.bfs.admin.ch/bfsstatic/dam/assets/8507281/master'
    path = DATA_DOWNLOADS_DIR / 'bfs_residence_work.xlsx'
    download_and_save(url, path, load=False)
    print(f"Loading {path}...", flush=True)
    sheet = pd.read_excel(path, sheet_name='Commune of residence perspect.',
                          header=header, skipfooter=4, usecols=usecols)
    return sheet

@cache
@cache_to_file(DATA_CACHE_DIR / 'bfs_residence_work_cols12568.df.csv')
def get_residence_work_cols12568():
    # (residence canton initial,
    #  residence commune number,
    #  work canton initial,
    #  work commune number,
    #  number of employed people)
    df = bfs_residence_work_xls(header=4, usecols=(1, 2, 5, 6, 8))
    df.columns = ('canton_home', 'number_home', 'canton_work', 'number_work', 'num_people')
    return df


# https://en.wikipedia.org/wiki/Cantons_of_Switzerland
CANTON_POPULATION = dict(zip(
    'ZH BE LU UR SZ OW NW GL ZG FR SO BS BL SH AR AI SG GR AG TG TI VD VS NE GE JU'.split(),
    [
        1520968,
        1034977,
        409557,
        36433,
        159165,
        37841,
        43223,
        40403,
        126837,
        318714,
        273194,
        200298,
        289527,
        81991,
        55234,
        16145,
        507697,
        198379 ,
        678207,
        276472,
        353343,
        799145,
        343955,
        176850,
        499480,
        73419,
    ]))

CANTON_KEYS_ALPHABETICAL = sorted(CANTON_POPULATION.keys())

CODE_TO_NAME = {
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

NAME_TO_CODE = {}
for code,name in CODE_TO_NAME.items():
    NAME_TO_CODE[name] = code

def fetch_openzh_covid_data(*, cache_duration=3600):
    """
    Returns a dictionary of lists {canton abbreviation: number of cases per day}.
    """
    url = 'https://raw.githubusercontent.com/daenuprobst/covid19-cases-switzerland/master/covid19_cases_switzerland_openzh.csv'
    path = DATA_DOWNLOADS_DIR / 'covid19_cases_switzerland_openzh.csv'

    raw = download_and_save(url, path, cache_duration=cache_duration)
    rows = raw.decode('utf8').split()
    cantons = rows[0].split(',')[1:-1]  # Skip the "Date" and "CH" cell.

    data = {canton: [] for canton in cantons}
    for day in rows[1:]:  # Skip the header.
        cells = day.split(',')[1:-1]  # Skip "Date" and "CH".
        assert len(cells) == len(cantons), (len(cells), len(cantons))

        for canton, cell in zip(cantons, cells):
            data[canton].append(float(cell or 'nan'))
    return data



COMMUTE_ADMIN_CH_CSV = DATA_FILES_DIR / 'switzerland_commute_admin_ch.csv'

@cache
@cache_to_file(DATA_CACHE_DIR / 'home_work_people.json',
               dependencies=[COMMUTE_ADMIN_CH_CSV])
def get_Cij_home_work_bfs():
    """
    Returns a dictionary
    {canton1: {canton2: number of commuters between canton1 and canton2, ...}, ...}.
    """
    #commute = swiss_mun.get_residence_work_cols12568()
    commute = get_residence_work_cols12568()

    Cij = {
        c1: {c2: 0 for c2 in CANTON_KEYS_ALPHABETICAL}
        for c1 in CANTON_KEYS_ALPHABETICAL
    }
    for home, work, num_people in zip(
            commute['canton_home'],
            commute['canton_work'],
            commute['num_people']):
        if home != work and work != 'ZZ':
            Cij[work][home] += num_people

    return Cij


def json_to_numpy_matrix(json, order):
    """Returns a json {'A': {'A': ..., ...}, ...} matrix as a numpy matrix.

    Arguments:
        json: A matrix in a JSON dictionary format.
        order: The desired row and column order in the output matrix.
    """
    assert len(order) == len(json), (len(order), len(json))
    out = np.zeros((len(order), len(order)))
    for index1, c1 in enumerate(order):
        for index2, c2 in enumerate(order):
            out[index1][index2] = json[c1][c2]
    return out


def get_Mij_numpy(canton_order):
    """Return the Mij numpy matrix using the data from bfs.admin.ch."""
    # NOTE: This is not the actual migration matrix!
    Cij = get_Cij_numpy(canton_order)
    return Cij + Cij.transpose()


def get_Cij_numpy(canton_order):
    """Return the mij numpy matrix using the data from bfs.admin.ch."""
    return json_to_numpy_matrix(get_Cij_home_work_bfs(), canton_order)


def get_shape_file():
    """
    Downloads and returns path to shape file with cantons.
    """
    zippath = DATA_DOWNLOADS_DIR / "swissBOUNDARIES3D.zip"
    download_and_save("https://shop.swisstopo.admin.ch/shop-server/resources/products/swissBOUNDARIES3D/download", zippath)


    shapefile = "BOUNDARIES_2020/DATEN/swissBOUNDARIES3D/SHAPEFILE_LV95_LN02/swissBOUNDARIES3D_1_3_TLM_KANTONSGEBIET"
    DATA_MAP_DIR = DATA_DOWNLOADS_DIR / "map"

    paths = extract_zip(zippath, shapefile, DATA_MAP_DIR)
    return os.path.splitext(paths[0])[0]

