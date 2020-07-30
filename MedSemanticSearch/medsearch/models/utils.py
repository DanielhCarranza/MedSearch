import numpy as np
import torch
from torch import Tensor
from typing import Union

def cosine_similarity(a:Union[Tensor,np.ndarray], b:Union[Tensor,np.ndarray])->Tensor:
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if isinstance(a, np.ndarray): a= torch.tensor(a)
    if isinstance(b, np.ndarray): b= torch.tensor(b)
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0, 1))
    return res