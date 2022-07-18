import os
import string
import nltk
import json
import torch
import faiss
import numpy as np
from typing import Dict, List, Tuple

from flask import Flask
from flask import jsonify, request

from langdetect import detect
from models.knrm import KNRM

app = Flask(__name__)
model_is_ready = False
index_is_ready = False


class Helper:
    def __init__(self):
        self.emb_path_glove = os.environ['EMB_PATH_GLOVE']
        self.vocab_path = os.environ['VOCAB_PATH']
        self.emb_path_knrm = os.environ['EMB_PATH_KNRM']
        self.mlp_path = os.environ['MLP_PATH']
        torch.set_grad_enabled(False)

    def prepare_model(self):
        self.model = KNRM(
            emb_path=self.emb_path_knrm,
            mlp_path=self.mlp_path
        )
        with open(self.vocab_path, 'r') as f_in:
            self.vocab = json.load(f_in)
        global model_is_ready
        model_is_ready = True

    def _hadle_punctuation(self, inp_str: str) -> str:
        inp_str = str(inp_str)
        for punct in string.punctuation:
            inp_str = inp_str.replace(punct, ' ')
        return inp_str

    def _simple_preproc(self, inp_str: str):
        base_str = inp_str.strip().lower()
        str_wo_punct = self._hadle_punctuation(base_str)
        return nltk.word_tokenize(str_wo_punct)

    def prepare_index(self, documents: Dict[str, str]):
        oov_val = self.vocab["OOV"]
        self.documents = documents
        idxs, docs = [], []
        for idx in documents:
            idxs.append(int(idx))
            docs.append(documents[idx])
        embeddings = []
        emb_layer = self.model.embeddings.state_dict()['weight']
        for d in docs:
            tmp_emb = [self.vocab.get(w, oov_val) for w in self._simple_preproc(d)]
            tmp_emb = emb_layer[tmp_emb].mean(dim=0)
            embeddings.append(np.array(tmp_emb))
        embeddings = np.array([embedding for embedding in embeddings]).astype(np.float32)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index = faiss.IndexIDMap(self.index)
        self.index.add_with_ids(embeddings, np.array(idxs))
        index_size = self.index.ntotal
        global index_is_ready
        index_is_ready = True

        return index_size

    def _text_to_token_ids(self, text_list: List[str]):
        tokenized = []
        for text in text_list:
            tokenized_text = self._simple_preproc(text)
            token_idxs = [self.vocab.get(i, self.vocab["OOV"]) for i in tokenized_text]
            tokenized.append(token_idxs)
        max_len = max(len(elem) for elem in tokenized)
        tokenized = [elem + [0] * (max_len - len(elem)) for elem in tokenized]
        tokenized = torch.LongTensor(tokenized)
        return tokenized

    def get_suggestion(self,
                       query: str, ret_k: int = 10,
                       ann_k: int = 100) -> List[Tuple[str, str]]:
        q_tokens = self._simple_preproc(query)
        vector = [self.vocab.get(tok, self.vocab["OOV"]) for tok in q_tokens]
        emb_layer = self.model.embeddings.state_dict()['weight']
        q_emb = emb_layer[vector].mean(dim=0).reshape(1, -1)
        q_emb = np.array(q_emb).astype(np.float32)
        _, I = self.index.search(q_emb, k=ann_k)
        cands = [(str(i), self.documents[str(i)]) for i in I[0] if i != -1]
        inputs = dict()
        inputs['query'] = self._text_to_token_ids([query] * len(cands))
        inputs['document'] = self._text_to_token_ids([cnd[1] for cnd in cands])
        scores = self.model(inputs)
        res_ids = scores.reshape(-1).argsort(descending=True)
        res_ids = res_ids[:ret_k]
        res = [cands[i] for i in res_ids.tolist()]
        return res

    def query_handler(self, inp):
        input_json = json.loads(inp.json)
        queries = input_json["queries"]
        lang_check = []
        suggestions = []
        for q in queries:
            is_en = detect(q) == "en"
            lang_check.append(is_en)
            if not is_en:
                suggestions.append(None)
                continue
            suggestion = self.get_suggestion(q)
            suggestions.append(suggestion)
        return suggestions, lang_check

    def index_handler(self, inp):
        input_json = json.loads(inp.json)
        documents = input_json["documents"]
        index_size = self.prepare_index(documents)
        return index_size


hlp = Helper()


@app.route('/ping')
def ping():
    if not model_is_ready:
        return jsonify(status="not ready")
    return jsonify(status="ok")


@app.route('/query', methods=['POST'])
def query():
    if not model_is_ready or not index_is_ready:
        return json.dumps({"status": "FAISS is not initialized!"})
    suggestions, lang_check = hlp.query_handler(request)

    return jsonify(suggestions=suggestions, lang_check=lang_check)


@app.route('/update_index', methods=['POST'])
def update_index():
    index_size = hlp.index_handler(request)

    return jsonify(status="ok", index_size=index_size)


hlp.prepare_model()
