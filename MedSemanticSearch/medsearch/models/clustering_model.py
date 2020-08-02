import numpy as np
from dataclasses import dataclass, field
from typing import Union, List, Tuple, Callable, Dict, Optional

from medsearch.models.base import TorchModelBase
from medsearch.datasets.dataset import SemanticCorpusDataset
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

class ClusteringModel(TorchModelBase):
    def __init__(self, 
                  dataset_cls:type=SemanticCorpusDataset, 
                  network_fn:Callable=SentenceTransformer, 
                  dataset_args:Dict=None, 
                  network_args:Dict=None):
        super().__init__(dataset_cls,None, network_fn, dataset_args, network_args)


    def word_embeddings(self,  corpus):
        self.embedder = lambda txt: np.array(self.network.encode(txt))
        self.corpus_embed = self.embedder(corpus)

    def get_similarity_vecs(self, n_clusters:int=5):
        clustering_model = KMeans(n_clusters=n_clusters)
        clustering_model.fit(self.corpus_embed)
        cluster_assignment = clustering_model.labels_ 
        return cluster_assignment
