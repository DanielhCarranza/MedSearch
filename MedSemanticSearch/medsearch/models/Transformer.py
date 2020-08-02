import os 
import json
import numpy as np
import logging
from typing import Optional, Union, List, Dict

from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig


class Transformer(nn.Module):
    def __init__(self, model_name_or_path:str, max_seq_length: int =128,  model_args:Dict={}, chache_dir: Optional[str]=None):

        super(Transformer, self).__init__()
        self.config_keys = ['max_seq_length']
        self.max_seq_length = max_seq_length

        config = AutoConfig.from_pretrained(model_name_or_path, **model_args, chache_dir=chache_dir)
        self.network = AutoModel.from_pretrained(model_name_or_path, config=config, chache_dir= chache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, chache_dir=chache_dir)

    def forward(self, features):
        out_states = self.network(**features)
        out_tokens = out_states[0]

        cls_tokens = out_tokens[:,0,:]
        features.update({'token_embeddings': out_tokens, 'cls_token_embeddings':cls_tokens, 
                     'attention_mask': features['attention_mask']})
        
        if self.network.config.output_hidden_states:
            all_layer_idx =2
            if len(out_states)<3:
                all_layer_idx=1
            hidden_states = out_states[all_layer_idx]
            features.update({'all_layer_embeddings':hidden_states})
        return features
    
    def get_word_embeddings(self)->int:
        return self.network.config.hidden_size
    
    def tokenize(self, text:str)->List[int]:
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
    
    def get_sentence_features(self, tokens: List[int], pad_seq_length:int):
        pad_seq_length = min(pad_seq_length, self.max_seq_length) + 3 # Add space for special tokens
        return self.tokenizer.prepare_for_model(tokens, max_seq_length=pad_seq_length, pad_to_max_length=True, return_tensors='pt', truncation=True)
    
    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}
    
    def save(self, output_path:str):
        self.network.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'transformer_config.json'),'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)
    
    @staticmethod
    def load(input_path:str):
        with open(os.path.join(input_path, 'transformer_config.json')) as fIn:
            config = json.load(fIn)
        return Transformer(model_name_or_path=input_path, **config)


