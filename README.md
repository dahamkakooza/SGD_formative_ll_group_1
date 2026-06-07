# SDG 3 Indicator Text Classification

Our group's solution for Assignment 2: classifying text documents according to the
indicators of Sustainable Development Goal 3 (SDG 3, "Good Health and Well-being").

A document can have zero, one, or several indicators at once, so this is a
multi-label classification problem. The main metric is Hamming Loss (lower is
better). We also report micro and macro F1, precision, and recall.

All the work is in one notebook: `notebooks/sdg3_classification.ipynb`. It runs
end to end on Google Colab and covers data loading, EDA, preprocessing, feature
engineering, ten experiments, results, and final predictions on the test set.

The full write-up (methodology, results, discussion) is in the report PDF
submitted on Canvas.

## What's in this repo

```
  data/
    Devex_train.csv             # labelled training data
    Devex_test_questions.csv    # unlabelled test data
  notebooks/
    sdg3_classification.ipynb   # the full pipeline
  requirements.txt
  README.md
```

When you run the notebook it creates an `artifacts/` folder for the plots, the
results table, the saved model, and the test predictions.

## Data

- `Devex_train.csv`: around 3,000 samples. Labels are in columns `Label 1` to
  `Label 10`. After cleaning there are 27 unique indicators.
- `Devex_test_questions.csv`: same text columns, no labels. This is what we
  predict on.

Both files are read with `latin-1` encoding because the raw text contains HTML.

## How to run (Google Colab)

1. Open `notebooks/sdg3_classification.ipynb` in Colab.
2. Put the two CSV files in `/content/data/`. The notebook looks for them there.
3. Run the cells from the top. The first cell installs the extra packages and the
   second downloads the NLTK resources. Colab may restart once after the install;
   if it does, run again from that cell.
4. Run the rest in order. This reproduces all tables, figures, and the final
   predictions file.

Experiment 7 uses GloVe vectors, which the notebook downloads automatically. If
GloVe is missing, that one experiment is skipped and everything else still runs.

## How to run (local)

```
python -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn joblib tqdm
pip install -r requirements.txt
jupyter notebook
```

The core packages above come pre-installed on Colab, so they are not pinned in
`requirements.txt` to avoid version conflicts. Locally the notebook reads the data
from the `data/` folder in this repo, so no path changes are needed.

## How predictions are made

After the experiments, the notebook picks the model with the lowest Hamming Loss,
runs it on the test set, and saves the result to `artifacts/test_predictions.csv`.
Each row has the sample's `Unique ID`, its `Type`, and the predicted indicators
spread across `Label 1` to `Label 10`, matching the training file format.

## Requirements

Core (pre-installed on Colab): pandas, numpy, scikit-learn, matplotlib, seaborn,
joblib, tqdm.

Extra (in `requirements.txt`): sentence-transformers, gensim, nltk,
beautifulsoup4, lxml.

## Authors

Group 1. Individual contributions are listed in the contribution tracker submitted
with the report.
