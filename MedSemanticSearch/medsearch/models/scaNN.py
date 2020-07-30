"""ScaNN Accelerate Vector Similarity Search by approximating the Maximum Inner Product Search 
        with an Anisotropic Loss function https://arxiv.org/abs/1908.10396"""
    
import os
import h5py 
import requests
import tempfile
from typing import Union, List, Tuple, Callable, Dict, Optional

import scann
import numpy as np 
from medsearch.models.base import ModelBase


class ScaNN(ModelBase):
    def __init__(self, dataset_cls:type, 
                network_fn:Callable=scann.ScannBuilder, 
                dataset_args:Dict=None, network_args:Dict=None):
        super().__init__(dataset_cls, network_fn, dataset_args, network_args)
        # self.dataset = dataset
        # self.queries = queries

    def normalized_dataset(self,dataset):
        normalized_dataset = dataset / np.linalg.norm(dataset, axis=1)[:,None] 
        return normalized_dataset

    def fit(self, normalized_dataset):
        self.searcher = ( self.network(normalized_dataset, 10, "dot_product")
            .tree(num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000)
            .score_ah(2, anisotropic_quantization_threshold=0.2).reorder(100).create_pybind())

    def search_single_query(self, query, **kwargs):
        neighbors, distances =  self.searcher.search(query, kwargs) 
        return neighbors, distances

    def search_batch_queries(self,queries, **kwargs):
        neighbors, distances = self.searcher.search_batched(queries, kwargs)
        return neighbors, distances

    def evaluate_similarity(self,true_neighbors, queries, leaves_to_search, reorder):
        neighbors, distances = self.searcher.search_batched(queries, 
                leaves_to_search=leaves_to_search, pre_reorder_num_neighbors=reorder)
        recall = self.compute_recall(neighbors, true_neighbors)
        print(f"Recall {recall} ")
        return recall 

    @staticmethod
    def compute_recall(neighbors, true_neighbors):
        total = 0
        for gt_row, row in zip(true_neighbors, neighbors):
            total += np.intersect1d(gt_row, row).shape[0]
        return total / true_neighbors.size
    

def get_glove_example():
    with tempfile.TemporaryDirectory() as tmp:
        response = requests.get("http://ann-benchmarks.com/glove-100-angular.hdf5")
        loc = os.path.join(tmp, "glove.hdf5")
        with open(loc, 'wb') as f:
            f.write(response.content)
        glove_h5py = h5py.File(loc)
        return glove_h5py

