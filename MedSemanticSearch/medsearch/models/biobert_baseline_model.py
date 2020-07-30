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


def run_test():
    bio_bert_models= {1:"emilyalsentzer/Bio_ClinicalBERT", 
                    2:"monologg/biobert_v1.1_pubmed", 
                    3:"allenai/biomed_roberta_base",
                    4:"gsarti/biobert-nli"}
    dataset_args = {'batch':1000}
    network_args = {'pretrained_model_name_or_path':bio_bert_models[1]}
    model = BioBertModel(dataset_args=dataset_args, network_args=network_args, tokenizer_args=network_args)
    data = model.data.load_one_batch()
    corpus = [(f'{t} <SEP> {a}')[:512] for t,a in zip(data.title, data.paperAbstract)]
    queries = ["breast cancer", "brain damage"]
    sentences, s = model.get_similarity_vecs(queries, corpus)
    print(f"queries: {queries}")
    for j, q in enumerate(queries):
        print(f"query: {q}")
        for i, st in enumerate((sentences[j])):
            print(f"similar paper {i}")
            print(f"{st}")
            print("-------------------------------------------")

if __name__ == "__main__":
    run_test()