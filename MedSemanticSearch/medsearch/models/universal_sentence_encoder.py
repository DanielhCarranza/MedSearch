import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub

import numpy as np
from dataclasses import dataclass, field
from medsearch.datasets.dataset import SemanticCorpusDataset
from typing import Union, List, Tuple, Callable, Dict, Optional

class UniversalSentenceEncoderModel():
    def __init__(self, dataset_cls:type=SemanticCorpusDataset, dataset_args:Dict=None ):
        if dataset_args is None: dataset_args={}
        self.data = dataset_cls(**dataset_args)
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.model = hub.load(module_url)
        self.batch_size = 16

    def embed(self, input):
        return self.model(input)

    def get_similarity_vecs(self, queries:Union[str, List[str]], corpus:List[str], topk=5):
        n_samples = len(corpus)
        emb = np.zeros([n_samples, 512])
        num_batches = n_samples // self.batch_size
        for i in range(num_batches + 1):
            start = self.batch_size * i
            end = (self.batch_size * i) + self.batch_size
            emb[start:end] = self.embed(corpus[start:end])
        emb_query = self.embed(queries)[0]
        ### TODO make a separete function
        input_matrix = np.vstack([[emb_query] * n_samples])
        results = np.dot(input_matrix, emb.T)[0]
        topk = results.argsort()[-topk:][::-1]
        scores =[str(s) for s in results[topk]]
        sentences = [corpus[idx] for idx in topk]
        return sentences, scores

def run_test():
    model = UniversalSentenceEncoderModel(dataset_args={'batch':1000})
    data = model.data.load_one_batch()
    corpus = [(f'{t} <SEP> {a}')[:512] for t,a in zip(data.title, data.paperAbstract)]
    queries = ["breast cancer"]
    sentences, scores = model.get_similarity_vecs(queries, corpus)

    print(f"Queries: {queries}")
    for i, (st, sc) in enumerate(zip(sentences,scores)):
        print(f"Similar paper {i} Score : {sc}")
        print(f"{st}")
        print(f"-------------------------------------")
if __name__ == "__main__":
    run_test()
