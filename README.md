# Wine Quality and IMDB - Deep Learning

Deep learning part of the PRDL project, with two case studies.

1. **Wine quality (notebooks 00-07).** The same problem as in the ML report
   ([Wine_quality_machine_learning_report](https://github.com/sandragc8/Wine_quality_machine_learning_report)):
   predicting the quality of red and white "Vinho Verde" wines from 11 physicochemical measurements,
   both as a regression problem (the quality score) and as a 3-class classification problem
   (Low / Medium / High), with feed-forward neural networks in PyTorch compared with the ML models.
2. **IMDB movie reviews (notebooks 08-11).** Sentiment analysis (positive / negative) to compare the
   basic architectures (Dense, CNN, LSTM, GRU, LSTM + attention, Transformer), with ablation studies,
   transfer learning from a self-supervised model (DistilBERT: zero-shot, feature extraction and
   fine-tuning) and two small autoregressive generators (an LSTM and a mini-GPT).

## Structure

```
data/
  raw/              original UCI files (red, white and the dataset description)
  raw/imdb/         IMDB reviews (train.parquet, test.parquet), downloaded from Hugging Face
  ml_results/       nested CV results of the ML report, used for the comparison
notebooks/
  utils.py                               data loading, plot style, MLP class and training loop
  imdb_utils.py                          IMDB tokenizer, vocabulary, the text networks and their training loop
  00_eda.ipynb                           exploratory analysis of the wine data
  01_mlp_regression_classification.ipynb MLP for both tasks, comparison with the ML models
  02_regularisation.ipynb                overfitting of a big network, early stopping, dropout, ...
  03_learning_rate_batch_size.ipynb      effect of the learning rate and the batch size
  04_class_weights_ensemble.ipynb        weighted cross-entropy and ensemble of 5 networks
  05_feature_importance.ipynb            permutation importance, MLP vs random forest
  06_multitask_learning.ipynb            one network with two heads (score + class / wine type)
  07_transfer_learning.ipynb             pre-train on white wines, fine-tune on red wines
  08_imdb_architectures.ipynb            IMDB EDA; Dense, CNN, LSTM and Transformer (3 seeds), word-order test
  09_imdb_ablations.ipynb                ablations: CNN kernels/pooling/length, padding, GRU, BiLSTM, LSTM + attention, positions
  10_imdb_transfer_learning.ipynb        DistilBERT: zero-shot, feature extraction and fine-tuning vs training from scratch
  11_imdb_text_generation.ipynb          autoregressive LSTM and mini-GPT that generate positive / negative reviews
results/
  *.csv             tables with the results of each notebook
  figures/          plots saved by the notebooks
```

## How to run

```
pip install -r requirements.txt
cd notebooks
jupyter lab
```

The wine notebooks are independent and can be run in any order (each one takes 1-3 minutes on a
laptop CPU). The IMDB notebooks need `transformers` and the IMDB files in `data/raw/imdb/`, and they
use the GPU of the laptop if it is available (Apple `mps`). Notebook 08 must be run first (09 and 10
reuse its results); on an Apple M1 notebook 08 takes about 30 minutes and 09-11 about 1-1.5 hours in
total. The data preparation is the same as in the ML report (red and white wines together,
duplicated rows removed, 5,320 wines) and it is done by `load_wine()` in `utils.py`. All the
random seeds are fixed to 42, and notebooks 01-06 use the same 5 outer folds as the ML report, so
the results can be compared directly.

## Main results

| Experiment | Result |
|---|---|
| MLP, regression | RMSE 0.695 (random forest of the ML report: 0.680) |
| MLP, classification | Macro F1 0.579 (random forest: 0.604) |
| Big network without regularisation | Train RMSE 0.14, test RMSE 0.845 (overfitting). Early stopping alone fixes most of it (0.699) |
| Learning rate / batch size | Same RMSE (~0.695) except learning rate 0.1 (0.759); the defaults (0.001, 64) are the fastest |
| Class weights | Recall of High 0.47 -> 0.77, but Medium 0.61 -> 0.38, so macro F1 goes down (0.592 -> 0.564) |
| Ensemble of 5 networks, regression | RMSE 0.690 vs 0.695 for a single network (better in all 5 folds) |
| Permutation importance | Alcohol is the most important variable for both models; the MLP depends much more on `is_red` |
| Multi-task network (score + class) | Same results as two separate networks with half of the parameters |
| Multi-task network (score + wine type, without `is_red`) | RMSE 0.691 vs 0.697 for the single-task network without `is_red` |
| Transfer white -> red, 25 red wines | Fine-tuning RMSE 0.751 vs 0.816 from scratch (only if each wine type is standardised with its own statistics) |

- The MLP is better than the linear models and similar to SVM / KNN, but the random forest is
  still the best model for this small tabular dataset.
- Transfer learning only helps when there are very few red wines; with a few hundred the network
  trained from scratch is as good.

IMDB (accuracy on the 25,000 test reviews unless indicated):

| Experiment | Result |
|---|---|
| Architectures (3 seeds) | Dense 0.860, CNN 0.863, LSTM 0.815 (unstable), Transformer 0.837 |
| Shuffled words | Dense unchanged, CNN -0.029, LSTM -0.025, Transformer only -0.003 |
| LSTM with patience 4 (3 seeds) | 0.856 ± 0.011 (padding processed as input: 0.819, GRU 0.859, BiLSTM 0.841) |
| LSTM + attention | 0.875, the best network with 256 words |
| Transformer without positional embeddings | 0.853 vs 0.831 with them |
| Input length (CNN) | 64 words 0.772, 256 words 0.866, 512 words 0.878 |
| DistilBERT, 100 labelled reviews (2,000 test reviews) | Frozen features 0.779, fine-tuning 0.722, from scratch 0.58-0.62, zero-shot 0.656 |
| DistilBERT fine-tuned, 2,000 labelled reviews | 0.866, same as the networks from scratch with 20,000 reviews |
| Text generation | Perplexity Mini-GPT 165 vs LSTM 221; temperature trades sentiment agreement for diversity |

## Data

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining
from physicochemical properties. Decision Support Systems, 47(4):547-553, 2009.
Available at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality).

A. L. Maas, R. E. Daly, P. T. Pham, D. Huang, A. Y. Ng and C. Potts. Learning word vectors for
sentiment analysis. ACL 2011. Downloaded from [Hugging Face (`stanfordnlp/imdb`)](https://huggingface.co/datasets/stanfordnlp/imdb)
and saved as `data/raw/imdb/train.parquet` and `test.parquet`.
