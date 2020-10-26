import os
import pathlib
import time
import urllib.request

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

