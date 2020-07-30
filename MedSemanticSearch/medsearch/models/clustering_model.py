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


def run_test():
    list_of_models:Dict = {1:'roberta-base-nli-stsb-mean-tokens',
                2:'bert-base-nli-stsb-mean-tokens'}
    model = ClusteringModel(
                        dataset_args={"batch":1000},
                        network_args={"model_name_or_path":list_of_models[1]})
    data = model.data.load_one_batch()
    corpus = [(f'{t} <SEP> {a}')[:512] for t,a in zip(data.title, data.paperAbstract)]

    model.word_embeddings(corpus)
    num_clusters=5
    cluster_assignment = model.get_similarity_vecs(num_clusters)
    clustered_sentences = [[] for i in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(corpus[sentence_id])

    for i, cluster in enumerate(clustered_sentences):
        print("============  Cluster {i+1} =========================")
        print(cluster[i])
        print("\n ===================================")


if __name__ == "__main__":
    run_test()