import argparse 
import logging
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from medsearch.models.base import TorchModelBase
from typing import Union, List, Tuple, Callable, Dict, Optional

from transformers import AutoTokenizer, AutoModel


bio_bert_models= {1:"emilyalsentzer/Bio_ClinicalBERT", 
                  2:"monologg/biobert_v1.1_pubmed", 
                  3:"allenai/biomed_roberta_base",
                  4:"gsarti/biobert-nli"}

def getBertBaselineModel():
    BERT_MODEL='bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    model = BertModel.from_pretrained(BERT_MODEL) 
    return tokenizer, model

def getSentenceModel():
    # 'roberta-base-nli-stsb-mean-tokens'
    model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
    # emb = np.array(model.encode(txt)) 
    return model  

def getBioBertModel(MODEL_TYPE):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE)
    model = AutoModel.from_pretrained(MODEL_TYPE)
    return tokenizer, model

def sentenceEmbedding(sentences):
    emb =[]
    for sentence in sentences:
        input_ids = torch.tensor(tokenizer.encode(sentence.lower(), 
                                add_special_tokens=True)[:512]).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_ids)[0]
            res = torch.mean(outputs, dim=1).detach().cpu().numpy()
        emb.append(res[0])
    return np.array(emb)

def get_scores(query, corpus, topk=5):
    emb_query = sentenceEmbedding(query)
    emb_corpus= sentenceEmbedding(corpus)
    results = cosine_similarity(emb_query, emb_corpus)
    topk = results.argsort()[-topk:][::-1]
    scores = results[topk]
    sentences = [corpus[idx] for idx in topk]
    return [str(s) for s in scores], sentences

class BertBaseModel(TorchModelBase):
    def __init__(self, dataset_cls:type, 
                  network_fn:Callable=BertModel.from_pretrained, 
                  dataset_args:Dict=None, network_args:Dict=None):
        super().__init__(dataset_cls, network_fn, dataset_args, network_args)
     



if __name__ == "__main__":
    tokenizer, model = getBertBaselineModel()