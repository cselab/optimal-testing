"""
This file provides access to all relevant data aobut Swiss cantons and COVID-19:
    - list of cantons and their population count
    - connectivity matrix between cantons
    - number of cases per day, for each canton
"""
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

#DATA_DOWNLOADS_DIR = "~/optimal-testing/covid19/epidemics/downloads/"

DATA_DOWNLOADS_DIR = pathlib.Path(__file__).parent.absolute() / "downloads"

DAY = datetime.timedelta(days=1)

def download(url):
    """Download and return the content of a URL."""
    print(f"[Epidemics] Downloading {url}... ", end="", flush=True)
    req = urllib.request.urlopen(url)
    data = req.read()
    print("Done.", flush=True)
    return data

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
    path = DATA_DOWNLOADS_DIR / "covid19_cases_switzerland_openzh.csv"

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

COMMUTE_ADMIN_CH_CSV = DATA_DOWNLOADS_DIR / "switzerland_commute_admin_ch.csv"

def get_residence_work_cols12568():
    # (residence canton initial,
    #  residence commune number,
    #  work canton initial,
    #  work commune number,
    #  number of employed people)
    url = 'https://www.bfs.admin.ch/bfsstatic/dam/assets/8507281/master'
    path = DATA_DOWNLOADS_DIR / "bfs_residence_work.xlsx"
    download_and_save(url, path, load=False)
    print(f"Loading {path}...", flush=True)
    sheet = pd.read_excel(path, sheet_name='Commune of residence perspect.',
                          header=4, skipfooter=4, usecols=(1, 2, 5, 6, 8))
    sheet.columns = ('canton_home', 'number_home', 'canton_work', 'number_work', 'num_people')
    return sheet

def get_Cij_home_work_bfs():
    """
    Returns a dictionary
    {canton1: {canton2: number of commuters between canton1 and canton2, ...}, ...}.
    """
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

def get_Mij_numpy(canton_order):
    """Return the Mij numpy matrix using the data from bfs.admin.ch."""
    # NOTE: This is not the actual migration matrix!

    json = get_Cij_home_work_bfs()

    assert len(canton_order) == len(json), (len(canton_order), len(json))
    Cij = np.zeros((len(canton_order), len(canton_order)))
    for index1, c1 in enumerate(canton_order):
        for index2, c2 in enumerate(canton_order):
            Cij[index1][index2] = json[c1][c2]

    return Cij + Cij.transpose()

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

name = ['AG','AI','AR','BE','BL','BS','FR','GE','GL','GR',\
        'JU','LU','NE','NW','OW','SG','SH','SO','SZ','TG',\
        'TI','UR','VD','VS','ZG','ZH']

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def prepareData(days = -1,country = False):    
    cantons = 26

    IR = fetch_openzh_covid_data()

    if days == -1:
        days = len(IR['TI'])

    data = np.zeros((cantons,days))

    for c in range(cantons):
        c_i = name[c]
        data[c,0] = IR[c_i][0]
        for d in range(1,days):
            data[c,d] = IR[c_i][d] - IR[c_i][d-1]   


    y = []
    if country == True:
        for d in range(days):
            tot = 0.0
            for c in range(cantons):
                if np.isnan(data[c,d]) == False:
                    tot += data[c,d]
            y.append(tot) 
        return y

    for c in range(cantons):
        d1 = np.copy(data[c,:])
        for d in range(days):
            if np.isnan(d1[d]) == False:
                y.append(c)
                y.append(d)
                y.append(d1[d])
    return y