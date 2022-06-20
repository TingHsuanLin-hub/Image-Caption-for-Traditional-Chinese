import torch
from typing import Optional, List
from torch import Tensor
from gensim.models import word2vec
import json
import os
import numpy as np

MAX_DIM = 256

def read_json(file_name):
    with open(file_name) as handle:
        out = json.load(handle)
    return out

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = [3, MAX_DIM, MAX_DIM]

        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

class Word2vec_addon(): # add encode_plus to class word2vec
    def __init__(self, word2vec_model):
        self.word2vec = word2vec.Word2Vec.load(word2vec_model)


    def encode_plus(self, caption, max_length, pad_to_max_length=False, return_attention_mask=False): # max_length是用來補足句子不夠長的部分
        padding = np.zeros((self.word2vec.vector_size,), dtype=np.float32) # used for padding, start_token, end_token
        start_token = self.word2vec.wv["*"]
        end_token = self.word2vec.wv["~"]
        input_ids = [start_token] + [self.word2vec.wv[word] for word in caption if word in self.word2vec.wv] + [end_token]
        ini_len = len(input_ids)
        
        attention_mask = None
        if pad_to_max_length:
            for i in range(max_length-ini_len):
                input_ids.append(padding)
            if return_attention_mask:
                attention_mask = [1]*ini_len + [0]*(max_length-ini_len)
        else:
            attention_mask = [1]*ini_len
        
        assert len(input_ids)== max_length
        return {'input_ids': input_ids, 'attention_mask':attention_mask}
