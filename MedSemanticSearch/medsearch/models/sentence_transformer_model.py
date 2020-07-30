import numpy as np
from dataclasses import dataclass, field
from medsearch.models.base import TorchModelBase
from medsearch.datasets.dataset import SemanticCorpusDataset
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import Union, List, Tuple, Callable, Dict, Optional

class SentenceTransformerModel(TorchModelBase):
    def __init__(self, 
                  dataset_cls:type=SemanticCorpusDataset, 
                  network_fn:Callable=SentenceTransformer, 
                  dataset_args:Dict=None, 
                  network_args:Dict=None):
        super().__init__(dataset_cls,None, network_fn, dataset_args, network_args)

    def word_embeddings(self, queries:Union[str,List[str]], corpus):
        self.embed = lambda txt: np.array(self.network.encode(txt))
        self.emb_queries = self.embed(queries)
        self.emb_corpus = self.embed(corpus)
        self.corpus = corpus

    def get_similarity_vecs(self, topk:int=10):
        results = cosine_similarity(self.emb_queries, self.emb_corpus)[0]
        topk = results.argsort()[-topk:][::-1]
        scores = [str(s) for s in results[topk]]
        sentences = [self.corpus[idx] for idx in topk]
        return  sentences, scores




def run_test():
    list_of_models:Dict = {1:'roberta-base-nli-stsb-mean-tokens',
                2:'bert-base-nli-stsb-mean-tokens'}
    model = SentenceTransformerModel(
                        dataset_args={"batch":1000},
                        network_args={"model_name_or_path":list_of_models[1]})
    data = model.data.load_one_batch()
    corpus = [(f'{t} <SEP> {a}')[:512] for t,a in zip(data.title, data.paperAbstract)]

    queries = ["breast cancer"]
    model.word_embeddings(queries, corpus)
    sentences, scores = model.get_similarity_vecs()

    print(f"queries: {queries}")
    for i, (st, sc) in enumerate(zip(sentences,scores)):
        print(f"similar paper {i} Score : {sc}")
        print(f"{st}")
        print(f"-------------------------------------")

if __name__ == "__main__":
    run_test()