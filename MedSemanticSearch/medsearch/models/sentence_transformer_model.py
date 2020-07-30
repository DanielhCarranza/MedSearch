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




def run_test():
    list_of_models:Dict = {1:'roberta-base-nli-stsb-mean-tokens',
                2:'bert-base-nli-stsb-mean-tokens'}
    model = SentenceTransformerModel(
                        dataset_args={"batch":1000},
                        network_args={"model_name_or_path":list_of_models[1]})
    data = model.data.load_one_batch()
    corpus = [(f'{t} <SEP> {a}')[:512] for t,a in zip(data.title, data.paperAbstract)]
    queries = ["breast cancer", 'brain damage', 'heart attack']
    model.word_embeddings(corpus)

    for query in queries:
        results = model.get_similarity_vecs(query)
        print(f"========== Queries: {query}  ================")
        for i, (st, sc) in enumerate(results):
            print(f"Similar paper {i} Score : {sc}")
            print(f"{corpus[st]}")
            print(f"-------------------------------------")

if __name__ == "__main__":
    run_test()