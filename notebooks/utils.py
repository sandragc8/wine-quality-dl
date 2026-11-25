"""Functions used by all the notebooks: data, plot style, the network and the training loop.

Notebook 01 explains every function step by step. The other notebooks just import them.
"""

import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
DATA_DIR = Path("../data")
RES_DIR = Path("../results")
FIG_DIR = RES_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- plot style (same as the ML report)
PLUM, PURPLE, LILAC = "#4A1942", "#7D3C98", "#C39BD3"
PINK, ROSE = "#E0719E", "#F5B7C8"
ORANGE, APRICOT = "#E8743B", "#F6A96B"
CREAM, INK = "#FDF1EC", "#3B2A3F"

PALETTE = [PURPLE, PINK, ORANGE, LILAC, APRICOT, PLUM, ROSE]
CLASS_ORDER = ["Low", "Medium", "High"]
CMAP_SEQ = LinearSegmentedColormap.from_list("seq", ["#FFFFFF", ROSE, PINK, PURPLE, PLUM])


def set_style():
    sns.set_theme(style="white", palette=PALETTE)
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#CBB8CF", "axes.labelcolor": INK, "axes.titlecolor": PLUM,
        "axes.titleweight": "semibold", "axes.titlesize": 12, "axes.grid": True,
        "grid.color": "#F1E4EC", "xtick.color": INK, "ytick.color": INK,
        "legend.frameon": False, "text.color": INK,
    })


def save(fig, name):
    fig.savefig(FIG_DIR / name)
    print("Figure saved ->", FIG_DIR / name)


# ---------------------------------------------------------------- data
PHYSCHEM = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
            "pH", "sulphates", "alcohol"]
FEATURES = PHYSCHEM + ["is_red"]


def load_wine():
    """Same preparation as in the ML report: join red and white, remove duplicates, add targets."""
    red = pd.read_csv(DATA_DIR / "raw" / "winequality-red.csv", sep=";")
    white = pd.read_csv(DATA_DIR / "raw" / "winequality-white.csv", sep=";")
    red["wine_type"] = "red"
    white["wine_type"] = "white"
    wine = pd.concat([red, white], ignore_index=True).drop_duplicates().reset_index(drop=True)
    wine["quality_class"] = pd.cut(wine["quality"], bins=[0, 5, 6, 10], labels=CLASS_ORDER)
    wine["class_id"] = wine["quality_class"].cat.codes.astype(int)   # Low=0, Medium=1, High=2
    wine["is_red"] = (wine["wine_type"] == "red").astype(int)
    return wine


def to_tensor(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32)


# ---------------------------------------------------------------- model
class MLP(nn.Module):
    """Feed-forward network: [Linear -> (BatchNorm) -> ReLU -> (Dropout)] x len(hidden) -> Linear."""

    def __init__(self, n_in, hidden=(64, 32), n_out=1, dropout=0.0, batch_norm=False):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_out)

    def forward(self, x):
        return self.head(self.body(x))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------- training loop
def train_model(model, X_train, Y_train, X_val, Y_val, loss_fn, val_loss_fn=None,
                lr=1e-3, weight_decay=0.0, batch_size=64, max_epochs=300, patience=20,
                params=None, seed=SEED):
    """Mini-batch training with Adam and early stopping on the validation loss.

    Y_train and Y_val are tuples of tensors, so the same loop works for one target or for
    several targets (multi-task). loss_fn(output, *targets) returns the loss of one batch.
    With patience=None there is no early stopping and the model of the last epoch is kept.
    """
    val_loss_fn = val_loss_fn or loss_fn
    params = model.parameters() if params is None else params
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(X_train, *Y_train), batch_size=batch_size,
                        shuffle=True, generator=generator)

    history = {"train": [], "val": []}
    best_val, best_state, wait = np.inf, None, 0
    for epoch in range(max_epochs):
        model.train()
        running = 0.0
        for xb, *yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), *yb)
            loss.backward()          # backpropagation
            optimizer.step()         # gradient descent step (Adam)
            running += loss.item() * len(xb)
        history["train"].append(running / len(X_train))

        model.eval()
        with torch.no_grad():
            val = val_loss_fn(model(X_val), *Y_val).item()
        history["val"].append(val)

        if val < best_val - 1e-5:
            best_val, best_state, wait = val, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if patience is not None and wait >= patience:
                break

    if patience is not None:
        model.load_state_dict(best_state)    # go back to the best epoch
    history["best_epoch"] = int(np.argmin(history["val"])) + 1
    return history
