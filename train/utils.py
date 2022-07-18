import torch
from typing import Dict, List, Union, Callable


class RankingDataset(torch.utils.data.Dataset):
    def __init__(self, index_pairs_or_triplets: List[List[Union[str, float]]],
                 idx_to_text_mapping: Dict[str, str], vocab: Dict[str, int], oov_val: int,
                 preproc_func: Callable, max_len: int = 30):
        self.index_pairs_or_triplets = index_pairs_or_triplets
        self.idx_to_text_mapping = idx_to_text_mapping
        self.vocab = vocab
        self.oov_val = oov_val
        self.preproc_func = preproc_func
        self.max_len = max_len

    def __len__(self):
        return len(self.index_pairs_or_triplets)

    def _tokenized_text_to_index(self, tokenized_text: List[str]) -> List[int]:
        """Transform tokenized text to list of indexes"""

        return [self.vocab.get(word, self.oov_val) for word in tokenized_text]

    def _convert_text_idx_to_token_idxs(self, idx: int) -> List[int]:
        """Transform text idx to tokenized word"""

        text = self.idx_to_text_mapping.get(str(idx))
        tokenized_text = self.preproc_func(text)
        return tokenized_text[:self.max_len]

    def __getitem__(self, idx: int):
        raise NotImplementedError


class TrainTripletsDataset(RankingDataset):
    def __getitem__(self, idx):

        sample = self.index_pairs_or_triplets[idx]
        tokenized_query = self._convert_text_idx_to_token_idxs(sample[0])
        tokenized_docs_1 = self._convert_text_idx_to_token_idxs(sample[1])
        tokenized_docs_2 = self._convert_text_idx_to_token_idxs(sample[2])
        query = self._tokenized_text_to_index(tokenized_query)
        document_1 = self._tokenized_text_to_index(tokenized_docs_1)
        document_2 = self._tokenized_text_to_index(tokenized_docs_2)

        return {'query': query, 'document': document_1},\
               {'query': query, 'document': document_2},  sample[3]


class ValPairsDataset(RankingDataset):
    def __getitem__(self, idx):

        sample = self.index_pairs_or_triplets[idx]
        tokenized_query = self._convert_text_idx_to_token_idxs(sample[0])
        tokenized_docs = self._convert_text_idx_to_token_idxs(sample[1])
        query = self._tokenized_text_to_index(tokenized_query)
        document = self._tokenized_text_to_index(tokenized_docs)
        return {'query': query, 'document': document}, sample[2]


def collate_fn(batch_objs: List[Union[Dict[str, torch.Tensor], torch.FloatTensor]]):
    max_len_q1 = -1
    max_len_d1 = -1
    max_len_q2 = -1
    max_len_d2 = -1

    is_triplets = False
    for elem in batch_objs:
        if len(elem) == 3:
            left_elem, right_elem, label = elem
            is_triplets = True
        else:
            left_elem, label = elem

        max_len_q1 = max(len(left_elem['query']), max_len_q1)
        max_len_d1 = max(len(left_elem['document']), max_len_d1)
        if len(elem) == 3:
            max_len_q2 = max(len(right_elem['query']), max_len_q2)
            max_len_d2 = max(len(right_elem['document']), max_len_d2)

    q1s = []
    d1s = []
    q2s = []
    d2s = []
    labels = []

    for elem in batch_objs:
        if is_triplets:
            left_elem, right_elem, label = elem
        else:
            left_elem, label = elem

        pad_len1 = max_len_q1 - len(left_elem['query'])
        pad_len2 = max_len_d1 - len(left_elem['document'])
        if is_triplets:
            pad_len3 = max_len_q2 - len(right_elem['query'])
            pad_len4 = max_len_d2 - len(right_elem['document'])

        q1s.append(left_elem['query'] + [0] * pad_len1)
        d1s.append(left_elem['document'] + [0] * pad_len2)
        if is_triplets:
            q2s.append(right_elem['query'] + [0] * pad_len3)
            d2s.append(right_elem['document'] + [0] * pad_len4)
        labels.append([label])
    q1s = torch.LongTensor(q1s)
    d1s = torch.LongTensor(d1s)
    if is_triplets:
        q2s = torch.LongTensor(q2s)
        d2s = torch.LongTensor(d2s)
    labels = torch.FloatTensor(labels)

    ret_left = {'query': q1s, 'document': d1s}
    if is_triplets:
        ret_right = {'query': q2s, 'document': d2s}
        return ret_left, ret_right, labels
    else:
        return ret_left, labels
