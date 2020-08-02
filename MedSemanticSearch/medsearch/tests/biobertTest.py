import unittest
from medsearch.models.biobert_model import BioBertModel


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
        print(f"Query: {q}")
        for i, st in enumerate((sentences[j])):
            print(f"similar paper {i}")
            print(f"{st}")
            print("-------------------------------------------")

if __name__ == "__main__":
    run_test()