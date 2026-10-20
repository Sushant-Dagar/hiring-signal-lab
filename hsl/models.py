"""Two resume-to-role matching models.

* Baseline: TF-IDF (unigrams + bigrams) into a multinomial logistic regression.
* PyTorch model: a fastText-style network. Each resume is a bag of hashed
  unigram and bigram ids, averaged through an EmbeddingBag, then a small
  hidden layer and a linear head over the roles. Trained with class-weighted
  cross-entropy and early stopping on validation macro-F1.

Both return a score matrix (resumes x roles) so they can be ranked and
compared on exactly the same footing.
"""

from __future__ import annotations

import time
import zlib

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch import nn


# --------------------------------------------------------------- baseline
class TfidfLogReg:
    name = "TF-IDF + Logistic Regression"

    def __init__(self, seed: int = 42):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60_000, sublinear_tf=True)
        self.clf = LogisticRegression(C=8.0, max_iter=2000, class_weight="balanced", random_state=seed)

    def fit(self, texts, y):
        self.clf.fit(self.vec.fit_transform(texts), y)
        return self

    def scores(self, texts) -> np.ndarray:
        return self.clf.predict_proba(self.vec.transform(texts))


# --------------------------------------------------------------- PyTorch MLP
class TorchTfidfMLP:
    """PyTorch MLP over the same TF-IDF features as the baseline.

    Sharing the features isolates the model choice: any difference against
    the logistic regression comes from the network, not from the text
    representation.
    """

    name = "PyTorch MLP on TF-IDF"

    def __init__(self, n_classes: int, seed: int = 42, hidden: int = 256, dropout: float = 0.5,
                 epochs: int = 30, batch_size: int = 64, lr: float = 1e-3, weight_decay: float = 1e-5,
                 patience: int = 4):
        self.n_classes, self.seed, self.hidden, self.dropout = n_classes, seed, hidden, dropout
        self.epochs, self.batch_size, self.lr, self.wd, self.patience = epochs, batch_size, lr, weight_decay, patience
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60_000, sublinear_tf=True)
        self.history: list[dict] = []

    @staticmethod
    def _dense(X, rows):
        return torch.from_numpy(X[rows].toarray().astype(np.float32))

    def fit(self, texts, y, val_texts, val_y, class_weights=None, log=print):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        Xtr = self.vec.fit_transform(texts).tocsr()
        Xva = self.vec.transform(val_texts).tocsr()
        y = np.asarray(y)
        d = Xtr.shape[1]
        self.model = nn.Sequential(nn.Dropout(0.2), nn.Linear(d, self.hidden), nn.ReLU(),
                                   nn.Dropout(self.dropout), nn.Linear(self.hidden, self.n_classes))
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        w = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)

        best, best_state, bad = -1.0, None, 0
        for ep in range(1, self.epochs + 1):
            t0 = time.time()
            self.model.train()
            order = rng.permutation(Xtr.shape[0])
            total = 0.0
            for i in range(0, len(order), self.batch_size):
                idx = order[i:i + self.batch_size]
                loss = loss_fn(self.model(self._dense(Xtr, idx)), torch.from_numpy(y[idx]))
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * len(idx)
            val_f1 = f1_score(val_y, self._scores(Xva).argmax(1), average="macro")
            self.history.append({"epoch": ep, "train_loss": total / Xtr.shape[0], "val_macro_f1": float(val_f1)})
            log(f"  epoch {ep:2d}  loss {total / Xtr.shape[0]:.4f}  val macro-F1 {val_f1:.4f}  ({time.time() - t0:.1f}s)")
            if val_f1 > best + 1e-4:
                best, bad = val_f1, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    log(f"  early stop: best val macro-F1 {best:.4f}")
                    break
        self.model.load_state_dict(best_state)
        self.best_epoch = int(np.argmax([h["val_macro_f1"] for h in self.history]) + 1)
        return self

    @torch.no_grad()
    def _scores(self, X) -> np.ndarray:
        self.model.eval()
        return np.vstack([torch.softmax(self.model(self._dense(X, np.arange(i, min(i + 512, X.shape[0])))), 1).numpy()
                          for i in range(0, X.shape[0], 512)])

    def scores(self, texts) -> np.ndarray:
        return self._scores(self.vec.transform(texts).tocsr())


# --------------------------------------------------------------- PyTorch fastText-style
N_BUCKETS = 2**20
MAX_TOKENS = 1500


def _hash(tok: str) -> int:
    return zlib.crc32(tok.encode()) % N_BUCKETS


def featurise(text: str) -> list[int]:
    words = text.split()[:MAX_TOKENS]
    ids = [_hash(w) for w in words]
    ids += [_hash(a + "_" + b) for a, b in zip(words, words[1:])]
    return ids or [0]


def _batch(feats: list[list[int]]):
    offsets = np.cumsum([0] + [len(f) for f in feats[:-1]])
    flat = np.fromiter((i for f in feats for i in f), dtype=np.int64)
    return torch.from_numpy(flat), torch.from_numpy(offsets.astype(np.int64))


class BagNet(nn.Module):
    def __init__(self, n_classes: int, dim: int = 96, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.emb = nn.EmbeddingBag(N_BUCKETS, dim, mode="mean", sparse=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, n_classes))
        nn.init.uniform_(self.emb.weight, -0.05, 0.05)

    def forward(self, flat, offsets):
        return self.head(self.emb(flat, offsets))


class TorchBagMatcher:
    name = "PyTorch EmbeddingBag network"

    def __init__(self, n_classes: int, seed: int = 42, epochs: int = 25, batch_size: int = 64,
                 lr_emb: float = 0.005, lr_head: float = 2e-3, patience: int = 4):
        self.n_classes, self.seed, self.epochs = n_classes, seed, epochs
        self.batch_size, self.lr_emb, self.lr_head, self.patience = batch_size, lr_emb, lr_head, patience
        self.history: list[dict] = []

    def fit(self, texts, y, val_texts, val_y, class_weights=None, log=print):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        Xtr = [featurise(t) for t in texts]
        Xva = [featurise(t) for t in val_texts]
        y = np.asarray(y)
        self.model = BagNet(self.n_classes)
        opt_sparse = torch.optim.SparseAdam(list(self.model.emb.parameters()), lr=self.lr_emb)
        opt_dense = torch.optim.AdamW(self.model.head.parameters(), lr=self.lr_head, weight_decay=1e-4)
        w = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        loss_fn = nn.CrossEntropyLoss(weight=w)

        best, best_state, bad = -1.0, None, 0
        for ep in range(1, self.epochs + 1):
            t0 = time.time()
            self.model.train()
            order = rng.permutation(len(Xtr))
            total = 0.0
            for i in range(0, len(order), self.batch_size):
                idx = order[i:i + self.batch_size]
                flat, off = _batch([Xtr[j] for j in idx])
                loss = loss_fn(self.model(flat, off), torch.from_numpy(y[idx]))
                opt_sparse.zero_grad(); opt_dense.zero_grad()
                loss.backward()
                opt_sparse.step(); opt_dense.step()
                total += loss.item() * len(idx)
            val_scores = self._scores(Xva)
            val_f1 = f1_score(val_y, val_scores.argmax(1), average="macro")
            self.history.append({"epoch": ep, "train_loss": total / len(Xtr), "val_macro_f1": float(val_f1)})
            log(f"  epoch {ep:2d}  loss {total / len(Xtr):.4f}  val macro-F1 {val_f1:.4f}  ({time.time() - t0:.1f}s)")
            if val_f1 > best + 1e-4:
                best, bad = val_f1, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    log(f"  early stop: best val macro-F1 {best:.4f}")
                    break
        self.model.load_state_dict(best_state)
        self.best_epoch = int(np.argmax([h["val_macro_f1"] for h in self.history]) + 1)
        return self

    @torch.no_grad()
    def _scores(self, feats) -> np.ndarray:
        self.model.eval()
        outs = []
        for i in range(0, len(feats), 512):
            flat, off = _batch(feats[i:i + 512])
            outs.append(torch.softmax(self.model(flat, off), 1).numpy())
        return np.vstack(outs)

    def scores(self, texts) -> np.ndarray:
        return self._scores([featurise(t) for t in texts])
