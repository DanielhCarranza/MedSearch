import argparse 
import logging
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from medsearch.models.base import TorchModelBase
from medsearch.datasets.dataset import SemanticCorpusDataset
from typing import Union, List, Tuple, Callable, Dict, Optional

class BioBertModel(TorchModelBase):
    def __init__(self, 
                  dataset_cls:type=SemanticCorpusDataset, 
                  tokenizer_cls:Callable=AutoTokenizer.from_pretrained, 
                  network_fn:Callable=AutoModel.from_pretrained, 
                  dataset_args:Dict=None, 
                  network_args:Dict=None,
                  tokenizer_args:Dict=None):
        super().__init__(dataset_cls,tokenizer_cls, network_fn, dataset_args, network_args, tokenizer_args)
        self.network.eval()

    def embed(self, sentences):
        emb =[]
        for sentence in tqdm(sentences): 
            input_ids = torch.tensor(self.tokenizer.encode(sentence.lower(), 
                                    add_special_tokens=True)[:512]).unsqueeze(0)
            with torch.no_grad():
                outputs = self.network(input_ids)[0]
                res = torch.mean(outputs, dim=1).detach().cpu().numpy()
            emb.append(res[0])
        return np.array(emb)       

    def get_similarity_vecs(self,queries, corpus, topk=5):
        emb_query = self.embed(queries)
        emb_corpus= self.embed(corpus)
        results = cosine_similarity(emb_query, emb_corpus)
        topk = results.argsort()[:,-topk:][::-1]
        scores = [str(s) for s in results[:,topk]]
        sentences = [np.asarray(corpus)[idx] for idx in topk]
        return  sentences, scores

