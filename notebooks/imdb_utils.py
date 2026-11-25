"""Functions used by the IMDB notebooks (08-10): data, vocabulary, the four networks and the training loop.

Notebook 08 explains every function step by step. The other notebooks just import them.
"""

import copy
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from utils import DATA_DIR, SEED

PAD, UNK = 0, 1
TOKEN_RE = re.compile(r"[a-z0-9']+|[!?]")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------- data
def load_imdb():
    """The 25,000 training and 25,000 test reviews of the IMDB dataset (label 1 = positive)."""
    train = pd.read_parquet(DATA_DIR / "raw" / "imdb" / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "raw" / "imdb" / "test.parquet")
    return train, test


def tokenize(text):
    """Lower case, remove the HTML line breaks and keep words, numbers, '!' and '?'."""
    return TOKEN_RE.findall(text.lower().replace("<br />", " "))


def build_vocab(token_lists, max_size=20000, min_freq=2):
    """Word -> index, using ONLY the training reviews. 0 is the padding and 1 the unknown words."""
    counts = Counter(t for tokens in token_lists for t in tokens)
    words = [w for w, c in counts.most_common(max_size - 2) if c >= min_freq]
    return {w: i + 2 for i, w in enumerate(words)}


def encode(token_lists, vocab, max_len=256, padding="post"):
    """Indexes truncated to max_len and padded with 0 at the end ("post") or at the beginning ("pre"),
    and the real length of each review."""
    X = np.full((len(token_lists), max_len), PAD, dtype=np.int64)
    lengths = np.zeros(len(token_lists), dtype=np.int64)
    for i, tokens in enumerate(token_lists):
        ids = [vocab.get(t, UNK) for t in tokens[:max_len]]
        if padding == "post":
            X[i, :len(ids)] = ids
        else:
            X[i, max_len - len(ids):] = ids
        lengths[i] = max(len(ids), 1)
    return torch.tensor(X), torch.tensor(lengths)


def prepare_imdb(max_len=256, padding="post", val_size=0.2):
    """The whole preparation of notebook 08 in one call: split, vocabulary and tensors (X, lengths, y)."""
    train_df, test_df = load_imdb()
    tok_train = [tokenize(t) for t in train_df.text]
    tok_test = [tokenize(t) for t in test_df.text]
    idx_tr, idx_val = train_test_split(np.arange(len(train_df)), test_size=val_size,
                                       stratify=train_df.label, random_state=SEED)
    vocab = build_vocab([tok_train[i] for i in idx_tr])
    X, L = encode(tok_train, vocab, max_len, padding)
    X_te, L_te = encode(tok_test, vocab, max_len, padding)
    y = torch.tensor(train_df.label.values, dtype=torch.float32)
    y_te = torch.tensor(test_df.label.values, dtype=torch.float32)
    return {"train": (X[idx_tr], L[idx_tr], y[idx_tr]), "val": (X[idx_val], L[idx_val], y[idx_val]),
            "test": (X_te, L_te, y_te), "vocab": vocab, "vocab_size": len(vocab) + 2,
            "idx_tr": idx_tr, "idx_val": idx_val, "train_df": train_df, "test_df": test_df}


def shuffle_words(X, lengths, seed=SEED):
    """Same reviews with the real words in a random order (the padding stays at the end)."""
    rng = np.random.default_rng(seed)
    Xs = X.clone()
    for i, n in enumerate(lengths.tolist()):
        Xs[i, :n] = X[i, rng.permutation(n)]
    return Xs


# ---------------------------------------------------------------- models
def masked_mean(h, x):
    """Average of h over the real words only (the padding positions are not counted)."""
    keep = (x != PAD).unsqueeze(-1).float()
    return (h * keep).sum(1) / keep.sum(1).clamp(min=1)


class DenseBag(nn.Module):
    """Embedding -> average of the word vectors -> dense layers. The word order is lost."""

    def __init__(self, vocab_size, emb_dim=128, hidden=64, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.mlp = nn.Sequential(nn.Linear(emb_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 1))

    def forward(self, x, lengths):
        return self.mlp(masked_mean(self.emb(x), x)).squeeze(1)


class TextCNN(nn.Module):
    """Embedding -> 1D convolutions with several kernel sizes -> pooling over time -> linear (Kim, 2014)."""

    def __init__(self, vocab_size, emb_dim=128, n_filters=100, kernel_sizes=(3, 4, 5),
                 dropout=0.5, pooling="max"):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.convs = nn.ModuleList(nn.Conv1d(emb_dim, n_filters, k) for k in kernel_sizes)
        self.pooling = pooling
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), 1)

    def forward(self, x, lengths):
        e = self.emb(x).permute(0, 2, 1)     # (batch, seq, emb) -> (batch, emb, seq): Conv1d wants channels first
        feats = [torch.relu(conv(e)) for conv in self.convs]
        if self.pooling == "max":
            pooled = [f.max(dim=2).values for f in feats]
        else:
            pooled = [f.mean(dim=2) for f in feats]
        return self.fc(self.drop(torch.cat(pooled, dim=1))).squeeze(1)


def last_real_state(out, lengths):
    """out[i, lengths[i] - 1, :] for every review i."""
    last = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(2))
    return out.gather(1, last).squeeze(1)


def reverse_real_words(e, lengths):
    """Reverse the real positions of every sequence and leave the padding at the end."""
    t = torch.arange(e.size(1), device=e.device).unsqueeze(0)
    idx = lengths.unsqueeze(1) - 1 - t
    idx = torch.where(idx >= 0, idx, t)
    return e.gather(1, idx.unsqueeze(-1).expand(-1, -1, e.size(2)))


class RNNClassifier(nn.Module):
    """Embedding -> LSTM or GRU -> last hidden state -> linear (non-autoregressive, one output per review).

    mode says how the padding is handled:
      "pack"   pack_padded_sequence: the RNN stops at the last real word of each review.
      "gather" the state out[i, length_i - 1] of the last real word. With post-padding it is the same
               as "pack" (the padding comes after it), but much faster. For the backward direction a
               second RNN reads every review with its real words in reverse order.
      "naive"  h_n after the whole padded sequence, so the padding is processed as real input.
    """

    def __init__(self, vocab_size, emb_dim=128, hidden=128, cell="lstm", bidirectional=False,
                 mode="gather", dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        two_rnns = bidirectional and mode == "gather"
        self.rnn = rnn(emb_dim, hidden, batch_first=True, bidirectional=bidirectional and not two_rnns)
        self.rnn_back = rnn(emb_dim, hidden, batch_first=True) if two_rnns else None
        self.bidirectional, self.mode = bidirectional, mode
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * (2 if bidirectional else 1), 1)

    def forward(self, x, lengths):
        e = self.drop(self.emb(x))
        if self.mode == "pack":
            e = pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, h_n = self.rnn(e)
        if self.mode == "gather":
            lengths = lengths.to(x.device)
            h = last_real_state(out, lengths)
            if self.rnn_back is not None:
                out_back, _ = self.rnn_back(reverse_real_words(e, lengths))
                h = torch.cat([h, last_real_state(out_back, lengths)], dim=1)
        else:
            if isinstance(h_n, tuple):       # LSTM returns (h_n, c_n), GRU only h_n
                h_n = h_n[0]
            # with two directions h_n[-2] is the forward one and h_n[-1] the backward one,
            # which finishes at the first word
            h = torch.cat([h_n[-2], h_n[-1]], dim=1) if self.bidirectional else h_n[-1]
        return self.fc(self.drop(h)).squeeze(1)


class LSTMAttention(nn.Module):
    """Embedding -> LSTM -> attention over the outputs of all the real words -> linear.

    Instead of using only the last state, every word t gets a score v^T tanh(W h_t); the softmax of the
    scores gives the weights alpha_t, and the review is represented by sum_t alpha_t h_t.
    """

    def __init__(self, vocab_size, emb_dim=128, hidden=128, attn_dim=64, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.rnn = nn.LSTM(emb_dim, hidden, batch_first=True)
        self.score = nn.Sequential(nn.Linear(hidden, attn_dim), nn.Tanh(), nn.Linear(attn_dim, 1, bias=False))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def attention(self, x):
        out, _ = self.rnn(self.drop(self.emb(x)))       # padding at the end: it does not change the real states
        scores = self.score(out).squeeze(-1).masked_fill(x == PAD, -1e9)
        alpha = torch.softmax(scores, dim=1)             # one weight per word, the padding gets 0
        return out, alpha

    def forward(self, x, lengths):
        out, alpha = self.attention(x)
        context = (alpha.unsqueeze(-1) * out).sum(1)
        return self.fc(self.drop(context)).squeeze(1)


class TransformerClassifier(nn.Module):
    """Embedding + learned positions -> Transformer encoder -> average of the outputs -> linear."""

    def __init__(self, vocab_size, max_len=256, d_model=128, n_heads=4, n_layers=2, d_ff=256,
                 dropout=0.1, positional=True):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d_model) if positional else None
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x, lengths):
        h = self.emb(x)
        if self.pos is not None:
            h = h + self.pos(torch.arange(x.size(1), device=x.device))
        h = self.encoder(h, src_key_padding_mask=(x == PAD))   # attention never looks at the padding
        return self.fc(masked_mean(h, x)).squeeze(1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------- training loop
bce = nn.BCEWithLogitsLoss()    # includes the sigmoid, so the models have no final activation


def predict_proba(model, X, lengths, batch_size=512):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb, lb = X[i:i + batch_size].to(DEVICE), lengths[i:i + batch_size]
            probs.append(torch.sigmoid(model(xb, lb)).cpu())
    return torch.cat(probs).numpy()


def evaluate(model, X, lengths, y, batch_size=512):
    """Mean BCE loss and accuracy on a whole set."""
    model.eval()
    total, correct = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb, lb = X[i:i + batch_size].to(DEVICE), lengths[i:i + batch_size]
            yb = y[i:i + batch_size].to(DEVICE)
            out = model(xb, lb)
            total += bce(out, yb).item() * len(xb)
            correct += ((out > 0).float() == yb).sum().item()
    return total / len(X), correct / len(X)


def train_text_model(model, train, val, lr=1e-3, batch_size=64, max_epochs=10, patience=2,
                     clip=1.0, seed=SEED, verbose=False):
    """Mini-batch training with Adam, gradient clipping and early stopping on the validation loss.

    train and val are tuples (X, lengths, y). The weights of the best epoch are restored at the end.
    """
    torch.manual_seed(seed)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(*train), batch_size=batch_size, shuffle=True,
                        generator=generator)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val, best_state, wait = np.inf, None, 0
    start = time.time()
    for epoch in range(max_epochs):
        model.train()
        running, correct = 0.0, 0
        for xb, lb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb, lb)
            loss = bce(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            running += loss.item() * len(xb)
            correct += ((out > 0).float() == yb).sum().item()
        history["train_loss"].append(running / len(train[0]))
        history["train_acc"].append(correct / len(train[0]))
        val_loss, val_acc = evaluate(model, *val)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if verbose:
            print(f"  epoch {epoch + 1}: train loss {history['train_loss'][-1]:.3f}, "
                  f"val loss {val_loss:.3f}, val acc {val_acc:.3f}")

        if val_loss < best_val - 1e-4:
            best_val, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    history["best_epoch"] = int(np.argmin(history["val_loss"])) + 1
    history["seconds"] = time.time() - start
    return history
