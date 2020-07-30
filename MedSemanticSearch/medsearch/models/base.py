"""Model class to be extended by specific types of models """
import json
from pathlib import Path
from datetime import datetime
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
        if network_fn is not None:
            self.network = network_fn(**network_args)

class TorchModelBase(ModelBase):
    def __init__(self, 
                  dataset_cls:type=None,
                  tokenizer_cls:Callable=None, 
                  network_fn:Callable=None, 
                  dataset_args:Dict=None, 
                  network_args:Dict=None,
                  tokenizer_args:Dict=None):
        super().__init__(dataset_cls, network_fn, dataset_args, network_args)

        if tokenizer_args is None: tokenizer_args={}
        if tokenizer_cls is not None:
            self.tokenizer = tokenizer_cls(**tokenizer_args)

    def evaluate(self):
        pass
    def loss(self):
        pass

    def metrics(self):
        pass

    @property
    def weights_filename(self):
        DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(DIRNAME/f'{self.name}_weights_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pt')

    def load_weights(self):
        pass

    def save_weights(self):
        pass

class TensorflowModelBase(ModelBase):

    def __init__(self,
                  dataset_cls:type, 
                  network_fn:Callable, 
                  dataset_args:Dict=None, 
                  network_args:Dict=None):
        super().__init__(dataset_cls, network_fn, dataset_args, network_args)

    def fit(self,dataset, bs, epochs, callbacks):
        pass
    def evaluate(self):
        pass
    def loss(self):
        pass

    def metrics(self):
        pass

    @property
    def weights_filename(self):
        DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(DIRNAME/f"{self.name}_weights.h5")

    def load_weights(self):
        pass

    def save_weights(self):
        pass