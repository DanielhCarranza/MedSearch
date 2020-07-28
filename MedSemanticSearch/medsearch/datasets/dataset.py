import os
import json
import pandas as pd
import numpy as np
import h5py
from pathlib import Path 
from typing import Union, List, Tuple
from similarity_abstract_search import utils
Path.ls = lambda x: list(x.iterdir())



class BaseDataset:
    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[3]/'Data/processed/SemanticScholarData'


class SemanticCorpusDataset(BaseDataset):
    def __init__(self, batch=64, data_path=None):
        self.batch= batch
        self.data_files = [fn for fn in SemanticCorpusDataset.data_dirname().ls() if fn.suffix=='.json']

    def load_one_batch(self):
        # Use generators
        data = pd.read_json(self.data_files[0]).set_index('EmbeddingID')
        self.data= data.loc[data.paperAbstract!=""]
        return self[0]

    def __getitem__(self,idx):
        begin = idx*self.batch
        end   = (idx + 1)*self.batch
        return self.data.iloc[begin:end,:]

    def __len__(self):
        return len(self.data)

    def show_data(self, idx=None):
        if idx is None:
            idx = np.random.randint(len(self))
        embed = self.data.iloc[idx,:]
        print(f'Title is : {embed["title"]}')
        print(f'Abstract is : {embed["paperAbstract"]}')
        print(f'Link is :  https://www.semanticscholar.org/paper/{embed["id"]}')

