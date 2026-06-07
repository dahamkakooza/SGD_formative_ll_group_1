# SDG 3 Indicator Text Classification

This repository contains our group's solution for Group Assignment 2: classifying
text documents (tenders, programs, reports, news, humanitarian and development
material) according to the indicators of Sustainable Development Goal 3 (SDG 3,
"Good Health and Well-being").

A single document can belong to zero, one, or several indicators at the same time,
so this is a multi-label text classification problem. The primary evaluation
metric required by the assignment is Hamming Loss (lower is better). We also report
micro and macro F1, precision, and recall to understand the trade-offs behind each
Hamming Loss number.

Everything lives in a single, fully documented notebook:
`notebooks/sdg3_classification.ipynb`. It runs end to end on Google Colab and
covers data loading, EDA, preprocessing, feature engineering, ten experiments,
results visualisation, and final inference on the test set.

## Dataset

The data is provided by the assignment and is included in `data/`:

- `data/Devex_train.csv` - the labelled training set (around 3,000 samples). The
  target is spread across the columns `Label 1` to `Label 10` (two extra label
  columns are fully empty and are dropped automatically in the notebook).
- `data/Devex_test_questions.csv` - the unlabelled test set we generate
  predictions for. It has the same text columns as the training file but no
  targets.

Both files are read with `latin-1` encoding because the raw text contains
embedded HTML and a few characters that break UTF-8 decoding.

After cleaning we end up with 27 unique SDG 3 indicators, 2,396 training samples,
599 validation samples, and 998 test samples.

## Repository structure

```
.
  data/
    Devex_train.csv
    Devex_test_questions.csv
  notebooks/
    sdg3_classification.ipynb     # the complete pipeline
  requirements.txt                # extra packages not pre-installed on Colab
  .gitignore
  README.md
```

When the notebook runs it creates an `artifacts/` folder for the plots, the
results table, the saved model, and the test predictions. That folder is not
committed (large binary model files are ignored in `.gitignore`).

## How to run on Google Colab (recommended)

This is the path the assignment asks for, and it needs almost no setup.

1. Upload the repository to Colab, or open the notebook directly from GitHub
   (File -> Open notebook -> GitHub tab).
2. Make sure the two CSV files are available at `/content/data/`. The notebook
   detects Colab automatically and looks for the data there. The simplest way is
   to upload the `data/` folder to `/content/data/`.
3. Run the cells from the top. The first cell installs the extra packages
   (`sentence-transformers`, `gensim`, `nltk`, `beautifulsoup4`, `lxml`) and the
   second downloads the small NLTK resources used for tokenisation and
   lemmatisation. Colab may restart the runtime once after the install; if it
   does, just run again from that cell.
4. For Experiment 7 (GloVe), the notebook downloads and unzips the GloVe 6B
   vectors into `/content/data/`. This is the only large download and it only
   matters for that one experiment. If GloVe is not present, the notebook skips
   Experiment 7 cleanly instead of crashing.

Run all the cells in order and you will reproduce every table, figure, and the
final predictions file.

## How to run locally

You can also run the notebook outside Colab.

1. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate
   ```
2. Install the core scientific stack (these come pre-installed on Colab, so they
   are intentionally left out of `requirements.txt` to avoid version clashes):
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn joblib tqdm
   ```
3. Install the extra packages used for preprocessing and embeddings:
   ```
   pip install -r requirements.txt
   ```
4. Launch Jupyter and open `notebooks/sdg3_classification.ipynb`:
   ```
   jupyter notebook
   ```

When running locally the notebook looks for the data in `../data` relative to the
notebook (that is, the `data/` folder in this repository), so no path changes are
needed if you keep the structure as is. GloVe is optional locally too; download
`glove.6B.100d.txt` and place it in `data/` if you want Experiment 7 to run.

## What the notebook does, section by section

1. Environment setup - installs packages and fixes a global random seed (42) so
   results are reproducible.
2. Data loading and schema inspection - reads both CSVs, finds the label columns,
   drops the empty ones, and builds the 27-class binary label matrix with
   `MultiLabelBinarizer`.
3. Exploratory data analysis - label frequency, number of labels per sample, a
   label co-occurrence heatmap, text length distribution, document type counts,
   and the most common tokens. These findings are what motivate the preprocessing
   and modelling choices later on.
4. Preprocessing pipeline - HTML stripping, lowercasing, removing URLs, emails and
   digits, tokenisation, stopword removal, and lemmatisation. Each step is
   controlled by a small config dictionary so an experiment can change one thing
   at a time.
5. Feature engineering - five text representations: Bag of Words, TF-IDF
   unigrams, TF-IDF unigrams plus bigrams, averaged GloVe vectors, and
   Sentence-Transformer (all-MiniLM-L6-v2) embeddings.
6. Experiments - ten experiments where each one changes a single variable
   relative to the previous one, so its effect on Hamming Loss is easy to read.
   Threshold tuning is done on a held-out half of the validation set and scored on
   the other half to avoid selection leakage.
7. Results comparison - the master results table, a Hamming Loss bar chart, a
   learning curve, and per-label F1 and precision-recall breakdowns for the best
   model.
8. Inference - applies the best model to the test set and writes predictions back
   out in the same wide `Label 1` to `Label 10` format as the training file.

## Experiments at a glance

The ten experiments, in the order they appear:

1. Bag of Words + Logistic Regression (baseline, default 0.5 threshold)
2. Bag of Words + Logistic Regression with balanced class weights
3. TF-IDF unigrams + Logistic Regression
4. TF-IDF unigrams and bigrams + Logistic Regression
5. TF-IDF unigrams + Linear SVM
6. TF-IDF unigrams + Logistic Regression + per-label threshold tuning
7. GloVe averaged embeddings + Logistic Regression
8. Sentence-Transformer embeddings + Logistic Regression
9. Sentence-Transformer embeddings + Linear SVM
10. Best dense model + per-label threshold tuning

Each experiment records what changed, why it was changed, and the resulting
Hamming Loss and F1 scores in a single results table. The full reasoning for each
step is written in the notebook markdown cells and discussed in the report.

Best result in our run: Experiment 10 (best dense model with per-label
thresholds) at a Hamming Loss of 0.0425. The threshold-tuned sparse model
(Experiment 6) and the TF-IDF Linear SVM (Experiment 5) were close behind. Exact
numbers can be reproduced by running the notebook.

## How predictions are generated

After the experiments, the notebook automatically picks the experiment with the
lowest Hamming Loss, saves it, and runs inference on the test set. The output is
written to `artifacts/test_predictions.csv`. Each row contains the test sample's
`Unique ID` and `Type`, followed by the predicted indicators laid out across
`Label 1` to `Label 10`, matching the format of the training file. Cells in
section 8 of the notebook print the prediction shape, the first few rows, and a
label count summary as a sanity check against the training distribution.

## Requirements

Core packages (pre-installed on Colab, install manually for local runs):
pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, tqdm.

Extra packages (in `requirements.txt`): sentence-transformers, gensim, nltk,
beautifulsoup4, lxml.

We intentionally do not pin versions for the core scientific stack so that the
notebook stays compatible with whatever Colab currently ships, which avoids the
binary incompatibility errors that come from forcing a different numpy or
scikit-learn build.

## Notes and limitations

- The validation split is a plain random 80/20 split. True stratification is not
  straightforward for multi-label targets, so we fix the seed instead to keep the
  split reproducible.
- The Sentence-Transformer embeddings are computed on the cleaned text and on the
  first part of each document, so very long documents are truncated. The report
  discusses how aggressive cleaning interacts with transformer embeddings.
- The test set has no ground-truth labels, so we cannot compute a test Hamming
  Loss directly. We evaluate on the held-out validation set and use the test set
  only for the final prediction file.

## Authors

Group 1. Individual contributions are tracked in the contribution tracker
included with the report submission.
