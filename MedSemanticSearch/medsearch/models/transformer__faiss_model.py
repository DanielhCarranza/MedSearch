import faiss 
import torch 

import numpy as np
from typing import Optional, Union, List, Dict, Callable

from medsearch.networks.Transformer import Transformer
from medsearch.models.base import TorchModelBase

class FaissTransformerModel(TorchModelBase):
    def __init__(self, dataset_cls:type, network_fn:Callable=Transformer, dataset_args:Dict=None, network_args:Dict=None):
        super().__init__(dataset_cls, None, network_fn, dataset_args, network_args)
        pass 
    def encode(self, document:List[str]):
        pass
    def search(self, query:Union[str, List[str]], topk:int=5)->List[str]:
        pass



