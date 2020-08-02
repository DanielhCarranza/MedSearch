import unittest
from medsearch.models.universal_sentence_encoder import UniversalSentenceEncoderModel

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
