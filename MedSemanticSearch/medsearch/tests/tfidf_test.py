import unittest
from medsearch.models.tfidf_model import TfidfModel

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
 