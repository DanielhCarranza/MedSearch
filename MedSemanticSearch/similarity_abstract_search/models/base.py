"""Model class to be extended by specific types of models """
import json
from pathlib import Path
from typing import Callable, Dict, Optional

DIRNAME = Path(__file__).parents[1].resolve()/'weights'

class ModelBase:
    def __init__(self, dataset_cls:type, network_fn:Callable, 
                dataset_args:dict=None, network_args:Dict=None):
                
        self.name = f"{self.__class__.__name__}_{dataset_cls.__name__}_{network_fn.__name__}"
        if dataset_args is None:
            dataset_args={}
        self.data = dataset_cls(**dataset_args)

        if network_args is None:
            network_args={}
        self.network = network_fn(**network_args)
    
    def load_weights(self, filename):
        pass

    def save_weights(self, obj, filename):
        with open(f'{filename}.json', 'w') as outfile:
            json.dump(obj, outfile)

class TorchModelBase(ModelBase):
    def model(self):
        pass

class TensorflowModelBase(ModelBase):
    def model(self):
        pass