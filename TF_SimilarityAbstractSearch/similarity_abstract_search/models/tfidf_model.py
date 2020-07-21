import numpy as np
from tqdm import tqdm
from typing import Union, List, Tuple, Callable, Dict, Optional
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from similarity_abstract_search.models.base import ModelBase
from similarity_abstract_search.datasets.dataset import SemanticCorpusDataset


class TfidfModel(ModelBase):
    def __init__(self, dataset_cls:type=SemanticCorpusDataset, 
                network_fn:Callable=TfidfVectorizer, 
                dataset_args:Dict=None, network_args:Dict=None):
        super().__init__(dataset_cls, network_fn, dataset_args, network_args)
        self.network=network_fn

    def fit(self,corpus:List[str]):
        """ Compute Tfidf features"""
        V = self.network(input='content',encoding='utf-8', 
                    decode_error='replace', strip_accents='unicode', 
                    lowercase=True, analyzer='word', stop_words='english', 
                    token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_-]+\b',ngram_range=(1,1), 
                    max_features=5000, norm='l2', use_idf=True, smooth_idf=True, 
                    sublinear_tf=True, max_df=1.0, min_df=3)
        X = np.asarray(V.fit_transform(corpus).astype(np.float32).todense())
        return X, V

    def dotSimilarity(self, X, nTake=50):
        S = X @ X.T
        simVec=np.argsort(S, axis=1)[:,:-nTake:-1]
        return simVec.tolist()  

    def svmSimilarity(self, X, nTake=40):
        n,_= X.shape
        IX = np.zeros((n,nTake), dtype=np.int64) 
        for i in tqdm(range(n)):
            # set all examples as negative except this one
            y = np.zeros(X.shape[0], dtype=np.float32) 
            y[i]=1
            clf = svm.LinearSVC(class_weight='balanced', 
                    verbose=False, max_iter=1000, tol=1e-4, C=0.1)
            clf.fit(X,y)
            simVec = clf.decision_function(X)
            ix = np.argsort(simVec)[:-nTake-1:-1]
            IX[i]=ix
        return IX.tolist()
    



