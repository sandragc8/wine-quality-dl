# %% [markdown]
# # 08 - Sentiment analysis of movie reviews: Dense, CNN, LSTM and Transformer
# IMDB dataset (25,000 training and 25,000 test reviews), binary classification (positive / negative)
#
# The wine dataset of notebooks 01-07 is a table: every wine is a row of 12 numbers without any order,
# so the only sensible network is a feed-forward one. To study the other basic architectures we need
# data with structure. Here we use the case study of the course: sentiment analysis of movie reviews,
# where the input is a sequence of words. The four networks start in the same way (each word index is
# replaced by a trainable vector of 128 dimensions with `nn.Embedding`) and differ in how they combine
# the word vectors of a review:
#
# * **Dense:** the average of the word vectors and two dense layers. The order of the words is lost.
# * **CNN:** 1D convolutions with kernels of 3, 4 and 5 words that slide along the review, followed by
#   max pooling over time. They detect local groups of words ("not good at all").
# * **LSTM:** reads the review word by word and keeps a memory; a dense layer is applied to the last
#   state (non-autoregressive: one prediction for the whole review).
# * **Transformer:** 2 encoder layers with self-attention (4 heads) and learned positional embeddings;
#   the outputs of all the words are averaged.
#
# All of them use the same vocabulary, the same training (Adam 0.001, batch 64, binary cross-entropy,
# at most 10 epochs, early stopping with patience 2 on the validation loss) and 3 random seeds.

# %%
import warnings
warnings.filterwarnings("ignore")

from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from utils import (set_style, save, SEED, RES_DIR, PURPLE, PINK, ORANGE, LILAC, PLUM, ROSE,
                   APRICOT, INK)
from imdb_utils import *

set_style()
np.random.seed(SEED)
torch.manual_seed(SEED)
SEEDS = [42, 43, 44]
MAX_LEN = 256
MODELS = ["Dense", "CNN", "LSTM", "Transformer"]
COLORS = dict(zip(MODELS, [LILAC, PINK, ORANGE, PURPLE]))
print("Device:", DEVICE)

# %% [markdown]
# ## 1. Data and exploratory analysis

# %%
train_df, test_df = load_imdb()
print(train_df.shape, test_df.shape)
print("Positive reviews in train / test:", train_df.label.mean(), test_df.label.mean())
print(train_df.text.iloc[3][:400])

# %%
tok_train = [tokenize(t) for t in train_df.text]
tok_test = [tokenize(t) for t in test_df.text]
len_train = np.array([len(t) for t in tok_train])
len_test = np.array([len(t) for t in tok_test])
print("Words per review (train): median", np.median(len_train), "mean", len_train.mean().round(1),
      "min", len_train.min(), "max", len_train.max())
for name, l in [("positive", len_train[train_df.label == 1]), ("negative", len_train[train_df.label == 0])]:
    print(f"  {name}: median {np.median(l):.0f}")
print(f"Reviews longer than {MAX_LEN} words (they will be truncated): {(len_train > MAX_LEN).mean():.1%}")

# %% [markdown]
# The vocabulary is built only with the training part (80% of the 25,000 training reviews), otherwise
# the validation and test words would leak into the model. We keep the 20,000 most frequent words;
# index 0 is the padding and index 1 the unknown words.

# %%
idx_tr, idx_val = train_test_split(np.arange(len(train_df)), test_size=0.2,
                                   stratify=train_df.label, random_state=SEED)
vocab = build_vocab([tok_train[i] for i in idx_tr])
VOCAB_SIZE = len(vocab) + 2
test_tokens = [t for toks in tok_test for t in toks]
coverage = np.mean([t in vocab for t in test_tokens])
n_distinct = len(Counter(t for i in idx_tr for t in tok_train[i]))
print(f"Distinct words in the training part: {n_distinct:,}; vocabulary size: {VOCAB_SIZE:,}")
print(f"Test words covered by the vocabulary: {coverage:.1%}")

# %% [markdown]
# Which words separate positive from negative reviews? For the words that appear at least 300 times in
# the training part we compute the log-ratio of their frequency in positive and negative reviews.

# %%
pos_counts = Counter(t for i in idx_tr if train_df.label.iloc[i] == 1 for t in set(tok_train[i]))
neg_counts = Counter(t for i in idx_tr if train_df.label.iloc[i] == 0 for t in set(tok_train[i]))
words = [w for w in vocab if pos_counts[w] + neg_counts[w] >= 300]
log_ratio = pd.Series({w: np.log((pos_counts[w] + 1) / (neg_counts[w] + 1)) for w in words}).sort_values()
top = pd.concat([log_ratio.head(10), log_ratio.tail(10)])
print(top.round(2))
print("'not':", round(log_ratio["not"], 2), " 'good':", round(log_ratio["good"], 2),
      " 'bad':", round(log_ratio["bad"], 2))

# %%
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), gridspec_kw=dict(width_ratios=[1.25, 1]))
ax = axes[0]
bins = np.arange(0, 1001, 25)
for lab, name, col in [(1, "Positive", PURPLE), (0, "Negative", PINK)]:
    ax.hist(np.clip(len_train[train_df.label == lab], 0, 1000), bins=bins, alpha=0.6, color=col, label=name)
ax.axvline(MAX_LEN, color=PLUM, ls="--", lw=1.5)
ax.text(MAX_LEN + 15, ax.get_ylim()[1] * 0.85,
        f"max length = {MAX_LEN}\n{(len_train > MAX_LEN).mean():.0%} of the reviews\nare truncated",
        color=PLUM, fontsize=9)
ax.set_xlabel("Words per review (values above 1,000 shown at 1,000)")
ax.set_ylabel("Number of reviews")
ax.set_title("Length of the training reviews")
ax.legend()
ax = axes[1]
ax.barh(top.index, top.values, color=[PINK if v < 0 else PURPLE for v in top.values])
ax.axvline(0, color=INK, lw=0.8)
ax.set_xlabel("log(frequency in positive / frequency in negative)")
ax.set_title("Most positive and negative words")
ax.tick_params(axis="y", labelsize=8.5)
ax.grid(axis="y", visible=False)
fig.tight_layout()
save(fig, "imdb_eda.png")
plt.show()

# %% [markdown]
# ## 2. From words to tensors
#
# Every review becomes a vector of 256 indexes (truncated or padded with 0 at the end). We also keep
# its real length, which the LSTM needs to know where the review ends.

# %%
X_all, L_all = encode(tok_train, vocab, MAX_LEN)
y_all = torch.tensor(train_df.label.values, dtype=torch.float32)
X_test, L_test = encode(tok_test, vocab, MAX_LEN)
y_test = torch.tensor(test_df.label.values, dtype=torch.float32)
train = (X_all[idx_tr], L_all[idx_tr], y_all[idx_tr])
val = (X_all[idx_val], L_all[idx_val], y_all[idx_val])
test = (X_test, L_test, y_test)
print("train", train[0].shape, "validation", val[0].shape, "test", test[0].shape)
print("First words:", tok_train[3][:8])
print("Indexes:    ", X_all[3, :8].tolist(), "... last positions:", X_all[3, -5:].tolist())

# %% [markdown]
# The tensor layout is the most common source of errors. The embedding gives (batch, sequence, channels),
# but `nn.Conv1d` expects (batch, channels, sequence), so the CNN permutes the last two axes. The LSTM
# and the Transformer use `batch_first=True` and work directly with (batch, sequence, channels).

# %%
xb = train[0][:4]
e = nn.Embedding(VOCAB_SIZE, 128, padding_idx=PAD)(xb)
print("indexes   ", tuple(xb.shape))
print("embedding ", tuple(e.shape), "(batch, sequence, channels)")
print("for Conv1d", tuple(e.permute(0, 2, 1).shape), "(batch, channels, sequence)")
print("after Conv1d with kernel 3:", tuple(nn.Conv1d(128, 100, 3)(e.permute(0, 2, 1)).shape))

# %% [markdown]
# ## 3. The four networks

# %%
def make_model(name, max_len=MAX_LEN):
    if name == "Dense":
        return DenseBag(VOCAB_SIZE)
    if name == "CNN":
        return TextCNN(VOCAB_SIZE)
    if name == "LSTM":
        return RNNClassifier(VOCAB_SIZE, mode="gather")
    return TransformerClassifier(VOCAB_SIZE, max_len)

emb_params = VOCAB_SIZE * 128
params = pd.DataFrame({name: {"total": count_parameters(make_model(name)),
                              "without embedding": count_parameters(make_model(name)) - emb_params}
                       for name in MODELS}).T
print(params)

# %% [markdown]
# **Padding in the LSTM.** The zero-padding is processed by the LSTM as if it were real input, so the
# last state `h_n` of a padded review is not the state after its last word. The standard solution is
# `pack_padded_sequence`, but on this computer it is 6-10 times slower. With the padding at the end and
# one direction, taking the output `out[i, length_i - 1]` of the last real word gives exactly the same
# result, because the padding comes after it. We check it with random weights:

# %%
torch.manual_seed(0)
xb, lb = val[0][:64], val[1][:64]
packed = RNNClassifier(VOCAB_SIZE, mode="pack").eval()
gather = RNNClassifier(VOCAB_SIZE, mode="gather").eval()
naive = RNNClassifier(VOCAB_SIZE, mode="naive").eval()
gather.load_state_dict(packed.state_dict())
naive.load_state_dict(packed.state_dict())
with torch.no_grad():
    print("max |pack - gather| =", (packed(xb, lb) - gather(xb, lb)).abs().max().item())
    print("max |pack - naive|  =", (packed(xb, lb) - naive(xb, lb)).abs().max().item())

# %% [markdown]
# ## 4. Training (3 seeds per network)
#
# The seed changes the initial weights, the dropout and the order of the mini-batches; the split and
# the vocabulary are always the same. Every model is also evaluated on the test reviews with their words
# in a random order, to see which networks use the order of the words.

# %%
X_test_shuffled = shuffle_words(X_test, L_test)
rows, curves, probs = [], {}, {}
for name in MODELS:
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = make_model(name)
        hist = train_text_model(model, train, val, seed=seed)
        p = predict_proba(model, X_test, L_test)
        p_shuf = predict_proba(model, X_test_shuffled, L_test)
        probs[(name, seed)] = p
        if seed == SEED:
            curves[name] = hist
        rows.append({"model": name, "seed": seed, "epochs": len(hist["val_loss"]),
                     "best epoch": hist["best_epoch"], "seconds": hist["seconds"],
                     "seconds per epoch": hist["seconds"] / len(hist["val_loss"]),
                     "val accuracy": max(hist["val_acc"]),
                     "test accuracy": accuracy_score(y_test, p > 0.5),
                     "test AUC": roc_auc_score(y_test, p),
                     "shuffled accuracy": accuracy_score(y_test, p_shuf > 0.5)})
        print(f"{name:12s} seed {seed}: best epoch {hist['best_epoch']}, "
              f"test acc {rows[-1]['test accuracy']:.4f}, shuffled {rows[-1]['shuffled accuracy']:.4f}, "
              f"{hist['seconds']:.0f} s")

runs = pd.DataFrame(rows)
runs.to_csv(RES_DIR / "imdb_architectures_runs.csv", index=False)
pd.DataFrame({f"{n}_{s}": p for (n, s), p in probs.items()}).to_csv(RES_DIR / "imdb_test_probs.csv", index=False)

# %%
summary = runs.groupby("model", sort=False).agg(
    test_accuracy=("test accuracy", "mean"), test_accuracy_std=("test accuracy", "std"),
    test_AUC=("test AUC", "mean"), shuffled_accuracy=("shuffled accuracy", "mean"),
    best_epoch=("best epoch", "mean"), seconds_per_epoch=("seconds per epoch", "mean"),
    seconds=("seconds", "mean"))
summary["parameters"] = params["total"]
summary["parameters without embedding"] = params["without embedding"]
summary.to_csv(RES_DIR / "imdb_architectures_summary.csv")
summary.round(4)

# %% [markdown]
# ## 5. Learning curves (seed 42)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
for name in MODELS:
    h = curves[name]
    ep = np.arange(1, len(h["val_loss"]) + 1)
    axes[0].plot(ep, h["train_loss"], color=COLORS[name], lw=1.6, ls="--")
    axes[0].plot(ep, h["val_loss"], color=COLORS[name], lw=2.2, label=name)
    axes[1].plot(ep, h["val_acc"], color=COLORS[name], lw=2.2, marker="o", ms=4, label=name)
axes[0].plot([], [], color=INK, ls="--", label="training (dashed)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Binary cross-entropy")
axes[0].set_title("Training (dashed) and validation (solid) loss")
axes[0].legend(fontsize=8.5)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Validation accuracy")
axes[1].set_title("Validation accuracy")
axes[1].legend(fontsize=8.5)
for ax in axes:
    ax.set_xticks(range(1, 11))
fig.tight_layout()
save(fig, "imdb_learning_curves.png")
plt.show()

# %% [markdown]
# ## 6. Test results and word order

# %%
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
ax = axes[0]
s = summary.loc[MODELS]
ax.bar(MODELS, s["test_accuracy"], yerr=s["test_accuracy_std"], color=[COLORS[m] for m in MODELS],
       edgecolor="white", error_kw=dict(ecolor=INK, lw=1, capsize=3))
for i, m in enumerate(MODELS):
    ax.text(i, s.loc[m, "test_accuracy"] + 0.004,
            f"{s.loc[m, 'test_accuracy']:.3f}\n{s.loc[m, 'seconds_per_epoch']:.0f} s/epoch",
            ha="center", fontsize=8.5)
ax.set_ylim(0.80, 0.92)
ax.set_ylabel("Test accuracy")
ax.set_title("Test accuracy (mean of 3 seeds)")
ax.grid(axis="x", visible=False)
ax = axes[1]
w = 0.38
xs = np.arange(len(MODELS))
ax.bar(xs - w / 2, s["test_accuracy"], w, color=[COLORS[m] for m in MODELS], label="original order")
ax.bar(xs + w / 2, s["shuffled_accuracy"], w, color=[COLORS[m] for m in MODELS], alpha=0.45,
       hatch="//", edgecolor="white", label="words shuffled")
for i, m in enumerate(MODELS):
    d = s.loc[m, "shuffled_accuracy"] - s.loc[m, "test_accuracy"]
    ax.text(i + w / 2, s.loc[m, "shuffled_accuracy"] + 0.004, f"{d:+.3f}", ha="center", fontsize=8.5)
ax.set_xticks(xs, MODELS)
ax.set_ylim(0.6, 0.92)
ax.set_ylabel("Test accuracy")
ax.set_title("Same test reviews with the words in a random order")
ax.legend(loc="lower left", fontsize=8.5)
ax.grid(axis="x", visible=False)
fig.tight_layout()
save(fig, "imdb_architectures.png")
plt.show()

# %% [markdown]
# ## 7. Accuracy by review length
#
# The reviews longer than 256 words are truncated, and the LSTM has to keep the information of the whole
# review in one state. We average the probabilities of the 3 seeds of each network and compute the
# accuracy for different review lengths (before truncation).

# %%
bins = [0, 100, 200, 256, 500, 3000]
labels = ["< 100", "100-200", "200-256", "256-500", "> 500"]
length_bin = pd.cut(len_test, bins=bins, labels=labels, right=True)
acc_len = pd.DataFrame({m: [accuracy_score(y_test[length_bin == b],
                                           np.mean([probs[(m, s)] for s in SEEDS], axis=0)[length_bin == b] > 0.5)
                            for b in labels] for m in MODELS}, index=labels)
acc_len["reviews"] = pd.Series(length_bin).value_counts().reindex(labels).values
acc_len.to_csv(RES_DIR / "imdb_accuracy_by_length.csv")
print(acc_len.round(3))

fig, ax = plt.subplots(figsize=(7.5, 4))
for m in MODELS:
    ax.plot(labels, acc_len[m], marker="o", lw=2, color=COLORS[m], label=m)
ax.set_xlabel("Words in the review (the model only sees the first 256)")
ax.set_ylabel("Test accuracy")
ax.set_title("Accuracy by review length (average of 3 seeds)")
ax.legend(fontsize=8.5)
fig.tight_layout()
save(fig, "imdb_accuracy_by_length.png")
plt.show()
