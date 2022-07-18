import os
import string
from collections import Counter
from typing import Dict, List, Tuple, Union

import nltk
import numpy as np
import math
import re
import pandas as pd
import torch

from utils import TrainTripletsDataset, ValPairsDataset, collate_fn
from models.knrm import KNRM

glue_qqp_dir = os.environ['GLUE_DIR']
glove_path = os.environ['GLOVE_PATH']

regex = re.compile('[%s]' % re.escape(string.punctuation))


class ModelTrainer:
    def __init__(self, glue_qqp_dir: str, glove_vectors_path: str,
                 min_token_occurancies: int = 1,
                 random_seed: int = 0,
                 emb_rand_uni_bound: float = 0.2,
                 freeze_knrm_embeddings: bool = True,
                 knrm_kernel_num: int = 21,
                 knrm_out_mlp: List[int] = [],
                 dataloader_bs: int = 1024,
                 train_lr: float = 0.001,
                 change_train_loader_ep: int = 10
                 ):
        self.glue_qqp_dir = glue_qqp_dir
        self.glove_vectors_path = glove_vectors_path
        self.glue_train_df = self.get_glue_df('train')
        self.glue_dev_df = self.get_glue_df('dev')
        self.dev_pairs_for_ndcg = self.create_val_pairs(self.glue_dev_df)
        self.min_token_occurancies = min_token_occurancies
        self.all_tokens = self.get_all_tokens(
            [self.glue_train_df, self.glue_dev_df], self.min_token_occurancies)
        self.random_seed = random_seed
        self.emb_rand_uni_bound = emb_rand_uni_bound
        self.freeze_knrm_embeddings = freeze_knrm_embeddings
        self.knrm_kernel_num = knrm_kernel_num
        self.knrm_out_mlp = knrm_out_mlp
        self.dataloader_bs = dataloader_bs
        self.train_lr = train_lr
        self.change_train_loader_ep = change_train_loader_ep

        self.model, self.vocab, self.unk_words = self.build_knrm_model()
        self.idx_to_text_mapping_train = self.get_idx_to_text_mapping(self.glue_train_df)
        self.idx_to_text_mapping_dev = self.get_idx_to_text_mapping(self.glue_dev_df)

        self.val_dataset = ValPairsDataset(self.dev_pairs_for_ndcg,
                                           self.idx_to_text_mapping_dev,
                                           vocab=self.vocab, oov_val=self.vocab['OOV'],
                                           preproc_func=self.simple_preproc)
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.dataloader_bs, num_workers=0,
            collate_fn=collate_fn, shuffle=False)

    def get_glue_df(self, partition_type: str) -> pd.DataFrame:
        assert partition_type in ['dev', 'train']
        glue_df = pd.read_csv(
            self.glue_qqp_dir + f'{partition_type}.tsv', sep='\t', error_bad_lines=False, dtype=object)
        glue_df = glue_df.dropna(axis=0, how='any').reset_index(drop=True)
        glue_df_fin = pd.DataFrame({
            'id_left': glue_df['qid1'],
            'id_right': glue_df['qid2'],
            'text_left': glue_df['question1'],
            'text_right': glue_df['question2'],
            'label': glue_df['is_duplicate'].astype(int)
        })
        return glue_df_fin

    @staticmethod
    def hadle_punctuation(inp_str: str) -> str:
        return regex.sub(' ', inp_str)

    def simple_preproc(self, inp_str: str) -> List[str]:
        inp_str = self.hadle_punctuation(inp_str).lower()
        tokenized = nltk.word_tokenize(inp_str)
        return tokenized

    @staticmethod
    def _filter_rare_words(vocab: Dict[str, int], min_occurancies: int) -> Dict[str, int]:
        """Remove rare words from counter"""

        filtered = {word: occurancies for word, occurancies in vocab.items() if vocab[word] >= min_occurancies}
        return filtered

    def get_all_tokens(self, list_of_df: List[pd.DataFrame], min_occurancies: int) -> List[str]:
        """Get all available tokens"""

        texts = []
        counts = Counter()
        for df in list_of_df:
            texts.extend(df.loc[:, ['text_left', 'text_right']].values.flatten().tolist())
        texts = set(texts)
        for text in texts:
            counts.update(self.simple_preproc(text))
        del texts
        filtered = self._filter_rare_words(counts, min_occurancies)
        return list(filtered.keys())

    def _read_glove_embeddings(self, file_path: str) -> Dict[str, List[str]]:
        """Get embeddings from file"""

        embeddings = {}
        with open(file_path) as f:
            for line in f:
                splitted = line[:-1].split(' ')
                embeddings[splitted[0]] = splitted[1:]
        return embeddings

    def create_glove_emb_from_file(self, file_path: str, inner_keys: List[str],
                                   random_seed: int, rand_uni_bound: float
                                   ) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
        """Create embeddings

        embeddings_array - all token embeddings
        word2ind - Dict mapping words to embedding indexes
        unk_words - words with generated embeddings
        """

        np.random.seed(random_seed)
        word_embeddings = self._read_glove_embeddings(file_path)
        unk_words = list(set(inner_keys).difference(set(word_embeddings.keys())))
        known_words = list(set(inner_keys).intersection(set(word_embeddings.keys())))
        embedding_size = len(word_embeddings[known_words[0]])
        embedding_array = np.zeros((len(inner_keys) + 2, embedding_size))

        word2ind = {"PAD": 0, "OOV": 1}
        unk_embedding = np.random.uniform(-rand_uni_bound, rand_uni_bound, embedding_size)
        embedding_array[1, :] = unk_embedding
        for index, word in enumerate(inner_keys, 2):
            embedding_array[index, :] = word_embeddings.get(word, np.random.uniform(-rand_uni_bound, rand_uni_bound, embedding_size))
            word2ind[word] = index
        unk_words += ["PAD", "OOV"]
        return embedding_array, word2ind, unk_words

    def build_knrm_model(self) -> Tuple[torch.nn.Module, Dict[str, int], List[str]]:
        emb_matrix, vocab, unk_words = self.create_glove_emb_from_file(
            self.glove_vectors_path, self.all_tokens, self.random_seed, self.emb_rand_uni_bound)
        torch.manual_seed(self.random_seed)
        knrm = KNRM(emb_matrix, freeze_embeddings=self.freeze_knrm_embeddings,
                    out_layers=self.knrm_out_mlp, kernel_num=self.knrm_kernel_num)
        return knrm, vocab, unk_words

    def sample_data_for_train_iter(self, inp_df: pd.DataFrame, seed: int
                                   ) -> List[List[Union[str, float]]]:
        np.random.seed(seed)
        groups = inp_df[['id_left', 'id_right', 'label']].groupby('id_left')
        pairs_w_labels = []
        all_right_ids = inp_df.id_right.values
        for id_left, group in groups:
            labels = group.label.unique()
            if len(labels) == 1:
                continue

            for label in labels:
                same_label_samples = group[group.label == label].id_right.values
                if label == 0 and len(same_label_samples) > 1:
                    sample = np.random.choice(same_label_samples, 2, replace=False)
                    pairs_w_labels.append([id_left, sample[0], sample[1], 0.5])
                elif label == 1:
                    less_label_samples = group[group.label < label].id_right.values
                    pos_sample = np.random.choice(same_label_samples, 1, replace=False)
                    if len(less_label_samples) > 0:
                        neg_sample = np.random.choice(less_label_samples, 1, replace=False)
                    else:
                        neg_sample = np.random.choice(all_right_ids, 1, replace=False)
                    pairs_w_labels.append([id_left, pos_sample[0], neg_sample[0], 1])

        return pairs_w_labels

    def create_val_pairs(self, inp_df: pd.DataFrame, fill_top_to: int = 15,
                         min_group_size: int = 2, seed: int = 0) -> List[List[Union[str, float]]]:
        inp_df_select = inp_df[['id_left', 'id_right', 'label']]
        inf_df_group_sizes = inp_df_select.groupby('id_left').size()
        glue_dev_leftids_to_use = list(
            inf_df_group_sizes[inf_df_group_sizes >= min_group_size].index)
        groups = inp_df_select[inp_df_select.id_left.isin(
            glue_dev_leftids_to_use)].groupby('id_left')

        all_ids = set(inp_df['id_left']).union(set(inp_df['id_right']))

        out_pairs = []

        np.random.seed(seed)

        for id_left, group in groups:
            ones_ids = group[group.label > 0].id_right.values
            zeroes_ids = group[group.label == 0].id_right.values
            sum_len = len(ones_ids) + len(zeroes_ids)
            num_pad_items = max(0, fill_top_to - sum_len)
            if num_pad_items > 0:
                cur_chosen = set(ones_ids).union(
                    set(zeroes_ids)).union({id_left})
                pad_sample = np.random.choice(
                    list(all_ids - cur_chosen), num_pad_items, replace=False).tolist()
            else:
                pad_sample = []
            for i in ones_ids:
                out_pairs.append([id_left, i, 2])
            for i in zeroes_ids:
                out_pairs.append([id_left, i, 1])
            for i in pad_sample:
                out_pairs.append([id_left, i, 0])
        return out_pairs

    def get_idx_to_text_mapping(self, inp_df: pd.DataFrame) -> Dict[str, str]:
        left_dict = (
            inp_df
            [['id_left', 'text_left']]
            .drop_duplicates()
            .set_index('id_left')
            ['text_left']
            .to_dict()
        )
        right_dict = (
            inp_df
            [['id_right', 'text_right']]
            .drop_duplicates()
            .set_index('id_right')
            ['text_right']
            .to_dict()
        )
        left_dict.update(right_dict)
        return left_dict

    def ndcg_k(self, ys_true: np.array, ys_pred: np.array, ndcg_top_k: int = 10) -> float:

        current_dcg = self._dcg_k(torch.Tensor(ys_true), torch.Tensor(ys_pred), ndcg_top_k)
        ideal_dcg = self._dcg_k(torch.Tensor(ys_true), torch.Tensor(ys_true), ndcg_top_k)
        return current_dcg / ideal_dcg

    @staticmethod
    def compute_gain(y_value: float, gain_scheme: str = 'exp2') -> float:
        if gain_scheme == "const":
            return y_value
        elif gain_scheme == "exp2":
            return 2 ** y_value - 1

    def _dcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor, k: int) -> float:
        _, indices = torch.sort(ys_pred, descending=True)
        sorted_true = ys_true[indices][:k].numpy()
        gain = self.compute_gain(sorted_true)
        discount = [math.log2(float(x)) for x in range(2, len(sorted_true) + 2)]
        discounted_gain = float((gain / discount).sum())
        return discounted_gain

    def valid(self, model: torch.nn.Module, val_dataloader: torch.utils.data.DataLoader) -> float:
        labels_and_groups = val_dataloader.dataset.index_pairs_or_triplets
        labels_and_groups = pd.DataFrame(labels_and_groups, columns=['left_id', 'right_id', 'rel'])

        all_preds = []
        for batch in (val_dataloader):
            inp_1, y = batch
            preds = model.predict(inp_1)
            preds_np = preds.detach().numpy()
            all_preds.append(preds_np)
        all_preds = np.concatenate(all_preds, axis=0)
        labels_and_groups['preds'] = all_preds

        ndcgs = []
        for cur_id in labels_and_groups.left_id.unique():
            cur_df = labels_and_groups[labels_and_groups.left_id == cur_id]
            ndcg = self.ndcg_k(cur_df.rel.values.reshape(-1), cur_df.preds.values.reshape(-1))
            if np.isnan(ndcg):
                ndcgs.append(0)
            else:
                ndcgs.append(ndcg)
        return np.mean(ndcgs)

    def train(self, n_epochs: int):
        opt = torch.optim.SGD(self.model.parameters(), lr=self.train_lr)
        criterion = torch.nn.BCELoss()
        for epoch in range(n_epochs):
            self.model.train()
            if epoch % 5 == 0:
                subset = self.sample_data_for_train_iter(self.glue_train_df, epoch)
                train_dataset = TrainTripletsDataset(subset,
                                                     self.idx_to_text_mapping_train,
                                                     vocab=self.vocab, oov_val=self.vocab['OOV'],
                                                     preproc_func=self.simple_preproc)

                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.dataloader_bs,
                                                               num_workers=0, collate_fn=collate_fn, shuffle=True)
            for batch in train_dataloader:
                inp1, inp2, y = batch
                preds = self.model(inp1, inp2)
                loss = criterion(preds, y)
                loss.backward()
                opt.step()
            if epoch > 3:
                with torch.no_grad():
                    self.model.eval()
                    ndcg = self.valid(self.model, self.val_dataloader)
                    print(f'Epoch {epoch} ndcg: {ndcg}')


if __name__ == '__main__':
    trainer = ModelTrainer(glue_qqp_dir, glove_path)
    trainer.train(5)
    torch.save(trainer.model.mlp.state_dict(), 'weights/MLP_weights')
