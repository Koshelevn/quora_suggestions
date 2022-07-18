from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import torch


class GaussianKernel(torch.nn.Module):
    """Class implementing single Gaussian Kernel"""

    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        forward_result = np.exp(-(x - self.mu)**2/(2*self.sigma**2))
        return forward_result


class KNRM(torch.nn.Module):
    def __init__(self, embedding_matrix: np.ndarray,
                 freeze_embeddings: bool,
                 kernel_num: int = 21,
                 sigma: float = 0.1,
                 exact_sigma: float = 0.001,
                 out_layers: List[int] = [10, 5],
                 mlp_path: Optional[str] = None):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=freeze_embeddings,
            padding_idx=0
        )

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()

        self.mlp = self._get_mlp()
        if mlp_path:
            self.mlp.load_state_dict(torch.load(mlp_path))

        self.out_activation = torch.nn.Sigmoid()

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        kernels = torch.nn.ModuleList()

        mus = [1.0]
        if self.kernel_num > 1:
            bin_size = 2.0 / (self.kernel_num - 1)
            mus.append(1 - bin_size / 2)
            for i in range(1, self.kernel_num - 1):
                mus.append(mus[i] - bin_size)
        mus = list(reversed(mus))
        sigmas = [self.sigma] * (self.kernel_num - 1) + [self.exact_sigma]

        for i in range(self.kernel_num):
            kernels.append(GaussianKernel(mus[i], sigmas[i]))
        return kernels

    def _get_mlp(self) -> torch.nn.Sequential:
        if not self.out_layers:
            return torch.nn.Sequential(OrderedDict({'Linear': torch.nn.Linear(self.kernel_num, 1)}))
        layers = OrderedDict()
        layers['Layer_K'] = torch.nn.Linear(self.kernel_num, self.out_layers[0])
        layers['ReLU'] = torch.nn.ReLU()

        for i in range(1, len(self.out_layers)):
            layers[f'Layer_{i}'] = torch.nn.Linear(self.out_layers[i-1], self.out_layers[i])
            layers[f'Relu_{i}'] = torch.nn.ReLU()
        layers['Last'] = torch.nn.Linear(self.out_layers[-1], 1)
        seq = torch.nn.Sequential(layers)
        return seq

    def forward(self, input_1: Dict[str, torch.Tensor], input_2: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)

        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)
        return out

    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:

        query_embeddings = self.embeddings(query)
        doc_embeddings = self.embeddings(doc)

        query_norm = query_embeddings / (query_embeddings.norm(p=2, dim=-1, keepdim=True) + 1e-13)
        doc_norm = doc_embeddings / (doc_embeddings.norm(p=2, dim=-1, keepdim=True) + 1e-13)

        return torch.FloatTensor(torch.bmm(query_norm, doc_norm.transpose(-1, -2)))

    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        km = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            km.append(K)

        # shape = [B, K]
        kernels_out = torch.stack(km, dim=1)
        return kernels_out

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        # shape = [Batch, Left, Embedding], [Batch, Right, Embedding]
        query, doc = inputs['query'], inputs['document']

        # shape = [Batch, Left, Right]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape = [Batch, Kernels]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape = [Batch]
        out = self.mlp(kernels_out)
        return out
