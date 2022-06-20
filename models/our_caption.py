import torch
from torch import nn
import torch.nn.functional as F

from .utils import NestedTensor, nested_tensor_from_tensor_list
from .transformer import build_transformer



class Caption(nn.Module):
    def __init__(self, transformer, hidden_dim, vocab_size):
        super().__init__()
        self.transformer = transformer
        self.mlp = MLP(hidden_dim, 512, vocab_size, 3)

    def forward(self,samples, target, target_mask):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        hs = self.transformer(samples.tensors, target, target_mask)
                              
        out = self.mlp(hs.permute(1, 0, 2))
        return out


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class samloss(nn.Module):
    def __init__(self):
        super(samloss, self).__init__()

    def forward(self,output, target):
        num = torch.sum(torch.multiply(output, target), 2)
        den = torch.sqrt(torch.multiply(torch.sum(output**2+1e-9, 2),torch.sum(target**2+1e-9, 2)))
        sam = torch.clip(torch.divide(num, den), -1, 1)
        sam = torch.mean(torch.arccos(sam))
        return sam


def build_model(config):
    transformer = build_transformer(config)

    model = Caption(transformer, config.hidden_dim, config.vocab_size)

    criterion = samloss()

    return model, criterion