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
