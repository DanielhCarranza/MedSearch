
import unittest
from medsearch.models.sentence_transformer_model import SentenceTransformerModel

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