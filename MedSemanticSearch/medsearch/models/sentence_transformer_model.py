from typing import Union, List, Tuple, Callable, Dict, Optional

import numpy as np
from medsearch.models.base import TorchModelBase
from medsearch.models.utils import cosine_similarity
from medsearch.datasets.dataset import SemanticCorpusDataset
from sentence_transformers import SentenceTransformer

class SentenceTransformerModel(TorchModelBase):
    def __init__(self, 
                  dataset_cls:type=SemanticCorpusDataset, 
                  network_fn:Callable=SentenceTransformer, 
                  dataset_args:Dict=None, 
                  network_args:Dict=None):
        super().__init__(dataset_cls,None, network_fn, dataset_args, network_args)

    def word_embeddings(self,  corpus:List[str]):
        self.embedder = lambda txt: np.array(self.network.encode(txt))
        self.corpus_embed = self.embedder(corpus)

    def get_similarity_vecs(self, query:Union[str,List[str]], topk:int=10):
        self.query_embed = self.embedder(query)
        scores = cosine_similarity(self.query_embed, self.corpus_embed)[0]
        results = zip(range(len(scores)), scores)
        results = sorted(results, key=lambda x: x[1], reverse=True)
        return results[:topk] 

