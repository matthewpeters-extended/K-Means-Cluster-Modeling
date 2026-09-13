# Setup

## Requirements

macOS with Homebrew Python 3.12. The system Python 3.9 at `/usr/bin/python3` is not used.

## Create the environment

The virtual environment already exists at `.venv`. To rebuild it from scratch:

```bash
cd ~/projects/financial-complaint-topic-modeling
rm -rf .venv
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
```

## Install dependencies

```bash
cd ~/projects/financial-complaint-topic-modeling && ./.venv/bin/pip install -r requirements.txt
```

## Download the NLTK corpora

```bash
cd ~/projects/financial-complaint-topic-modeling && ./.venv/bin/python -m nltk.downloader stopwords punkt punkt_tab wordnet omw-1.4 vader_lexicon
```

## Fetch the data

```bash
cd ~/projects/financial-complaint-topic-modeling && ./.venv/bin/python scripts/fetch_data.py
```

This writes `data/raw/complaints_stratified.parquet`,
`data/raw/complaints_natural.parquet` and `data/raw/fetch_metadata.json`. None of them are
committed to git. The script is the reproducible record of how the corpus was built.

### If it stops partway

It will not lose anything. Every stratum is checkpointed to `data/raw/_shards` the moment it
arrives, so if the run is interrupted, rate limited or cancelled, you simply run the same
command again and it picks up from the last completed stratum. Nothing already fetched is
fetched twice.

The API rate limits aggressive callers, so the script pauses one second between calls by
default. The full pull is 288 requests and takes roughly eight minutes. If you still get
rate limited, slow it down further:

```bash
cd ~/projects/financial-complaint-topic-modeling && ./.venv/bin/python scripts/fetch_data.py --sleep 2
```

To throw away the checkpoints and pull everything again from scratch:

```bash
cd ~/projects/financial-complaint-topic-modeling && ./.venv/bin/python scripts/fetch_data.py --fresh
```

To preview what it would pull without writing anything:

```bash
cd ~/projects/financial-complaint-topic-modeling && ./.venv/bin/python scripts/fetch_data.py --dry-run
```

## Register the Jupyter kernel

```bash
cd ~/projects/financial-complaint-topic-modeling && ./.venv/bin/python -m ipykernel install --user --name fintech-topics --display-name "Python 3.12 (fintech-topics)"
```

## Lock the environment

Once everything runs, freeze it so the results are reproducible:

```bash
cd ~/projects/financial-complaint-topic-modeling && ./.venv/bin/pip freeze > requirements-lock.txt
```

## Build the modelling corpus

```bash
cd ~/projects/financial-complaint-topic-modeling && ./.venv/bin/python scripts/build_corpus.py
```

Cleans both corpora and writes `data/processed/corpus_stratified.parquet` and
`corpus_natural.parquet`, plus a reduction report to `reports/`.

## Audit the data and draw the figures

```bash
cd ~/projects/financial-complaint-topic-modeling && ./.venv/bin/python scripts/audit_data.py
```

```bash
cd ~/projects/financial-complaint-topic-modeling && ./.venv/bin/python scripts/make_eda_figures.py
```

## Run the tests

```bash
cd ~/projects/financial-complaint-topic-modeling && ./.venv/bin/python -m pytest tests/ -q
```

## Build the feature matrices

```bash
cd ~/projects/financial-complaint-topic-modeling && ./.venv/bin/python scripts/build_features.py
```

Splits the corpus into train and test stratified by product, fits every vectoriser on the
training split alone, and writes the matrices to `data/processed/features` along with the
split itself. Comparison tables land in `reports/`.

## Run the clustering

```bash
cd ~/projects/financial-complaint-topic-modeling && ./.venv/bin/python scripts/run_clustering.py
```

Sweeps K from 2 to 20 across four representations and saves the fitted labels for K = 2 and
K = 8 so later phases do not refit. Takes roughly two minutes.

## Evaluate against the ground truth labels

```bash
cd ~/projects/financial-complaint-topic-modeling && ./.venv/bin/python scripts/evaluate_clusters.py
```

Scores every clustering against the bureau's product and issue labels, on train and on the
held out test split, with a majority baseline and a permuted control beside every number.
