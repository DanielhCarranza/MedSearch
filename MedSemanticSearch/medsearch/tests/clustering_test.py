
import unittest
from medsearch.models.clustering_model import ClusteringModel

def run_test():
    list_of_models:Dict = {1:'roberta-base-nli-stsb-mean-tokens',
                2:'bert-base-nli-stsb-mean-tokens'}
    model = ClusteringModel(
                        dataset_args={"batch":1000},
                        network_args={"model_name_or_path":list_of_models[1]})
    data = model.data.load_one_batch()
    corpus = [(f'{t} <SEP> {a}')[:512] for t,a in zip(data.title, data.paperAbstract)]

    model.word_embeddings(corpus)
    num_clusters=5
    cluster_assignment = model.get_similarity_vecs(num_clusters)
    clustered_sentences = [[] for i in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(corpus[sentence_id])

    for i, cluster in enumerate(clustered_sentences):
        print("============  Cluster {i+1} =========================")
        print(cluster[i])
        print("\n ===================================")


if __name__ == "__main__":
    run_test()