import gc 
import os
import re
import json 
import hashlib
import numpy as np
import pandas as pd 
import h5py
from tqdm import tqdm
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from typing import Union, Tuple, List
from urllib.request import urlopen, urlretrieve


def compute_sha256(filename: Union[Path, str]):
    """Return SHA256 checksum of a file."""
    with open(filename, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

### Processing 
open_raw_data_json = lambda j : json.loads(json.loads(j))

def get_raw_json_data(filename: Union[Path, str]):
  f = open(filename)
  jsonlist = f.readlines()
  df = pd.DataFrame(map(open_raw_data_json, jsonlist))
  del jsonlist, f
  gc.collect()
  return df

def get_paper_set(file: Union[Path, str])->set:
  file=str(file)
  if file.endswith('.txt'):
    ps = set(open(file).read().replace('"',"").splitlines())
  else: print('File not in the correct format')
  return ps

def save_dict2json(filename, dictObj):
  with open(f'{filename}.json', 'w') as outfile:
    json.dump(dictObj, outfile)

def load_json(filename):
  with open(filename,'r') as f:
    data = json.load(f)
  return data

def saveEmbedIDs(citesIDs:np.array, paperIDs:np.array, filename:str):
  h5f = h5py.File(f'{filename}.h5', 'w')
  dt = h5py.special_dtype(vlen=np.dtype('int32'))
  h5f.create_dataset('cites', data=citesIDs, dtype=dt, compression='gzip', compression_opts=9, chunks=True, maxshape=(None,) )
  h5f.create_dataset('paper', data=paperIDs, compression='gzip', compression_opts=9, chunks=True, maxshape=(None,) )
  h5f.close()

def loadH5file(filename):
  h5file = h5py.File(f'{filename}.h5','r')
  return h5file