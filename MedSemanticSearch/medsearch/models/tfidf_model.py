import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Union, List, Tuple, Callable, Dict, Optional

from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from medsearch.models.base import ModelBase
from medsearch.datasets.dataset import SemanticCorpusDataset


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

    def dotSimilarity(self, X:np.ndarray, nTake:int=50)->List[float]:
        S = X @ X.T
        simVec=np.argsort(S, axis=1)[:,:-nTake:-1]
        return simVec.tolist()  

    def svmSimilarity(self, X:np.ndarray, nTake:int=40)->List[float]:
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


    def build_search_index(self,data:pd.DataFrame, v:np.ndarray):

        # construct a reverse index for suppoorting search
        vocab = v.vocabulary_
        idf = v.idf_
        punc = "'!\"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'" # removed hyphen from string.punctuation
        trans_table = {ord(c): None for c in punc}

        def makedict(s, forceidf=None):
            words = set(s.lower().translate(trans_table).strip().split())
            words = set(w for w in words if len(w) > 1 and (not w in ENGLISH_STOP_WORDS))
            idfd = {}
            for w in words: # todo: if we're using bigrams in vocab then this won't search over them
                if forceidf is None:
                    if w in vocab:
                        idfval = idf[vocab[w]] # we have a computed idf for this
                    else:
                        idfval = 2.0 # some word we don't know; assume idf 1.0 (low)
                else:
                    idfval = forceidf
                idfd[w] = idfval
            return idfd

        def merge_dicts(dlist:List)->Dict:
            m = {}
            for d in dlist:
                for k, v in d.items():
                    m[k] = m.get(k,1) + v
            return m

        dict_title = data.title.apply(lambda s: makedict(s, forceidf=11))
        dict_summary = data['paperAbstract'].apply(makedict)#.to_dict()
        search_dict = [merge_dicts([t,s]) for t,s in zip(dict_title, dict_summary)]
        return search_dict

def run_test(save_dicts:bool=True):
    model = TfidfModel(dataset_args={"batch":5000})
    df = model.data.load_one_batch()
    corpus = [f'{t} <SEP> {a}' for t,a in zip(df.title, df.paperAbstract)]
    X,V = model.fit(corpus)
    IX = model.svmSimilarity(X)
    search = model.build_search_index(df, V)
    if save_dicts:
        model.save_weights(search, model.data.data_dirname().parent/'search')
        model.save_weights(IX, model.data.data_dirname().parent/'sim_vecs')

if __name__ == "__main__":
    run_test()
    


