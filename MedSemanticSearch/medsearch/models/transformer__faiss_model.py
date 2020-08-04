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
            
    def encode(self, document:List[str])->torch.Tensor:
        tokens = self.network.tokenize(document)
        embed  = self.network(**tokens)[0].detach().squeeze()
        return torch.mean(embed, dim=0) # Average Vector
    
    def load_word_vectors(input_path:str):
        pass

    def indexing(self):
        word_vectors = [self.encode(doc) for doc in self.documents]
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.network.get_word_embeddings_dim()))
        self.index.add_with_ids(np.array([t.numpy() for t in word_vectors]), np.array(range(0, len(self.documents))))

    def search(self, query:Union[str, List[str]], topk:int=5)->List[str]:
        query_embed = self.encode(query).unsqueeze(0).numpy()
        res = self.index.search(query_embed, k=topk)
        scores = res[0][0]
        results = [self.document[_id] for _id in res[1][0]]
        return list(zip(results, scores))




