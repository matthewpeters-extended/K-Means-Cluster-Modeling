# PLAN: Financial Complaint Topic Modelling

Unsupervised topic modelling over consumer finance complaint narratives, using TF IDF and
CountVectorizer features with KMeans clustering, benchmarked against the real product labels
that ship with the data.

Status: complete. All ten phases run. Results in `README.md`, the build narrative in
`WALKTHROUGH.md`, and per phase findings in `docs/`.
Created 2026 09 12, completed 2026 09 13.

## 1. What this project is

Take a large corpus of free text complaints that real people filed against banks, credit card
issuers, credit bureaus, debt collectors and money transfer services. Clean the text, vectorise
it, cluster it with KMeans, and read the clusters back as named complaint themes. Then check
honestly whether those discovered themes line up with the categories the regulator already
assigns by hand.

The resume value is not "I ran KMeans". It is that this corpus comes with ground truth labels,
so the clustering can actually be scored instead of only admired.

## 2. Reference solution and attribution

The starting specification is the public repository
`carlosrod723/NLP-KMeans-Topic-Modeling`, which clusters Vodafone customer tweets with
CountVectorizer plus KMeans and visualises each cluster as a word cloud. That repository is
credited explicitly in the README and in `docs/sources.md`.

We are not copying it. We are porting the method to a finance corpus and repairing the
evaluation gaps listed in section 7.

## 3. The brief we are building to

From the task description:

* Clean the text: strip symbols, numbers and short words
* Tokenise and count individual words
* Vectorise with both TF IDF and CountVectorizer
* Use NLTK, scikit learn and wordcloud
* Apply unsupervised KMeans clustering
* Run two centroid counts, K = 2 and K = 8, compare them and interpret what each cluster means
* Visualise with word clouds

All of that is in scope and will be delivered. Sections 6 and 7 add the rigour on top.

### One honest adjustment

The brief describes clusters as emotional states such as happy, neutral, sad and angry. That
framing fits a general tweet stream. It does not fit a complaints database, where every single
document is by definition a grievance, so a sentiment axis would collapse to "negative" almost
everywhere and the clusters would be uninterpretable as moods.

So we split the idea into two axes and report both:

* Theme axis, from KMeans. What is the complaint about. Billing, credit report errors, debt
  collection harassment, mortgage servicing, fraud and unauthorised charges, and so on.
* Intensity axis, from NLTK VADER sentiment scoring applied separately. How angry is the
  writer. This gets reported per theme, so we can say which complaint themes carry the most
  hostile language.

That preserves the intent of the brief and produces a result that is defensible.

## 4. Dataset

Chosen source: the Consumer Financial Protection Bureau Consumer Complaint Database.

* Public United States federal government data, no account, no API key, no licence fee
* Queried through a documented JSON API, so acquisition is scripted and reproducible
* Free text field `complaint_what_happened` is the corpus to cluster
* Ships with hand assigned `product`, `sub_product` and `issue` labels, which become the
  evaluation ground truth
* Also carries company, state, date received, submission channel and company response, which
  supports segmentation beyond the text itself

Verified against the live API on 2026 09 12. For complaints received in calendar 2024 that
carry a narrative, the totals are:

* Credit reporting or other personal consumer reports: 631,757
* Debt collection: 70,424
* Credit card: 35,515
* Checking or savings account: 29,533
* Mortgage: 12,235
* Money transfer, virtual currency, or money service: 10,255
* Student loan: 8,041
* Vehicle loan or lease: 7,565
* Payday loan, title loan, personal loan, or advance loan: 5,221
* Prepaid card: 3,714
* Debt or credit management: 1,562
* Total with narrative in 2024: 815,822

Full detail and the alternatives considered are in `docs/sources.md`.

### Sampling decision

That distribution is 77 percent credit reporting. If we sample at random the clusters will all
be credit reporting and the exercise is dead on arrival. So:

* Pull a stratified sample, capped per product per month, across September 2023 to
  December 2024
* Target roughly 20,000 narratives, deliberately flattened across products at 115 per product
  per month over 16 months
* Keep a second natural distribution sample of the same size, unflattened, and report how much
  worse clustering behaves on it

That contrast is itself a finding worth writing up.

Why the window starts in September 2023. The bureau renamed and resplit its product
categories in August 2023, so any window reaching further back mixes two incompatible labelling
schemes. September 2023 is the first clean month. The full finding, the evidence and the
reasoning behind not attempting to map the old scheme forward are in `docs/data_defects.md`
under D1.

Why 20,000 and not more. The reference solution clustered 16,194 documents, so this lands in
the same order of magnitude and the comparison stays fair. Our narratives average around 1,300
characters against a tweet limit of 280, so the corpus is roughly five times the text volume
per document even at similar row counts. KMeans on a sparse matrix this size fits in memory and
a full sweep of K = 2 to 20 runs in seconds rather than minutes, which matters because we run
that sweep many times across two vectorisers, two samples and three algorithms. The fetch takes
288 API requests and a few minutes. If the clusters turn out to be unstable at this size, the
sample is a command line flag and we scale it up rather than guessing upward first.

## 5. Environment and structure

* Python 3.12 from Homebrew at `/opt/homebrew/opt/python@3.12/bin/python3.12`
* Virtual environment at `.venv`, already created
* Dependencies in `requirements.txt`, with a `pip freeze` lock written once the work runs

Layout:

* `PLAN.md` this file
* `README.md` written last, with measured numbers only
* `WALKTHROUGH.md` the narrative walk through of what was done and why
* `data/raw` API pulls plus fetch metadata, not committed
* `data/processed` cleaned corpus, not committed
* `docs/` sources, setup, and one findings file per phase
* `notebooks/NN_*.ipynb` exploratory notebooks
* `reports/` metric tables as CSV
* `reports/figures/` all plots and word clouds
* `scripts/` runnable pipeline steps
* `src/` importable helpers
* `tests/` unit tests for the cleaning and metric code

## 6. Phases

### Phase 0. Setup
Create the folder, the virtual environment and the dependency list. Done.

### Phase 1. Acquisition
`scripts/fetch_data.py` walks the API product by product and month by month, writes
`data/raw/complaints.parquet` plus `data/raw/fetch_metadata.json` recording the exact query
parameters, the row count per stratum, the fetch timestamp and a checksum. Anyone can rerun it
and get the same corpus.

### Phase 2. Data defects audit
Before any modelling, write `docs/data_defects.md` covering:

* Redaction artefacts. The bureau masks names, dates and account numbers as runs of X. These
  are the single highest frequency tokens in the corpus and will dominate every cluster if left
  in place.
* Exact duplicates and near duplicates. Credit repair firms file templated narratives in bulk.
  Exact deduplication is not enough, so add a near duplicate pass over hashed token shingles.
* Length outliers. Some narratives are a single sentence, some are many pages.
* Empty and boilerplate only narratives after cleaning.
* Class imbalance, quantified.

### Phase 3. Exploratory analysis
Notebook `01_eda.ipynb`. Label distribution, narrative length distribution, vocabulary growth,
token frequency before and after stopword removal, complaint volume over time, top companies,
geographic spread. Figures to `reports/figures`. Findings to `docs/eda_findings.md`.

### Phase 4. Cleaning and tokenisation
`src/preprocess.py`, unit tested in `tests/`:

* Strip the X redaction runs
* Strip URLs, email addresses, currency symbols, punctuation and digits
* Lowercase
* Drop tokens of two characters or fewer
* NLTK stopwords plus a custom finance stoplist built from the max_df pass
* Lemmatise with the NLTK WordNet lemmatiser
* Deduplicate on cleaned text, then near deduplicate
* Write `data/processed/corpus.parquet`

### Phase 5. Vectorisation
Build both representations on the same cleaned corpus so they are directly comparable:

* CountVectorizer, unigrams, with min_df and max_df frequency gates
* TfidfVectorizer, same vocabulary gates, with and without bigrams

Record vocabulary size, matrix shape and sparsity for each. This is the part of the brief that
asks for both, and we will report which one produced better separated clusters rather than
assuming.

### Phase 6. Clustering
* Split into train and test. Fit the vectoriser and KMeans on train only, then assign test
  documents. The reference solution fits on everything, which cannot generalise.
* Deliver the two centroid counts the brief asks for, K = 2 and K = 8, with side by side
  interpretation of every cluster.
* Add the sweep the brief does not ask for, K = 2 to 20, scoring inertia for the elbow,
  silhouette, Davies Bouldin and Calinski Harabasz.
* Report whether the metric optimum agrees with K = 8. If it does not, say so plainly.
* Apply truncated SVD before clustering as a latent semantic variant and compare.

### Phase 7. Evaluation, the part that makes this a portfolio piece
Because the data carries real labels we can score the clusters instead of eyeballing them:

* Cluster purity against `product` and against `issue`
* Adjusted Rand index and normalised mutual information
* A contingency table of cluster against product, written to `reports/`
* Two baselines reported in the same breath as any headline number. The majority class
  baseline, which on the natural distribution 2024 slice is 77.4 percent for credit reporting.
  And a random assignment control at the same K, to prove the clustering beats shuffling.
* The same evaluation on the natural distribution sample and the stratified sample, to show
  what imbalance does to unsupervised methods.

### Phase 8. Method comparison
KMeans against Non negative Matrix Factorisation and against Latent Dirichlet Allocation on the
same matrices, scored the same way. The reference repository argues for KMeans without testing
the alternatives. We test them.

### Phase 9. Visualisation
* Word clouds per cluster, as the brief requires, for both K = 2 and K = 8
* A top terms table per cluster by centroid weight, because word clouds show frequency but
  hide the actual ranking
* Two dimensional projection of the document matrix with TSNE, coloured by cluster and coloured
  by true product label, side by side
* Cluster size bar chart and the K sweep metric curves
* VADER sentiment intensity distribution per theme

### Phase 10. Write up
`WALKTHROUGH.md` first, then `README.md` last with measured numbers only and no placeholders.
Include what did not work. Then initialise git and push.

## 7. Defects we inherit from the reference solution and will fix

1. No quantitative selection of K. The reference compares two values by eye. We run a full
   sweep with four metrics.
2. No evaluation against any ground truth. Our corpus has labels, so we report purity, adjusted
   Rand index and normalised mutual information.
3. No baseline of any kind. We always state the majority class rate and a random control next
   to the result.
4. Raw counts only, no TF IDF comparison, justified after the fact rather than tested. We build
   both and compare.
5. Fits on the entire corpus with no held out set. We split train and test.
6. Deduplication on exact string match only, which misses templated filings. We add near
   duplicate detection.
7. Word clouds as the only evidence of cluster quality. We add ranked term tables and metrics.
8. Data loaded from a private Google Drive, so nobody can reproduce it. Ours is fetched by a
   script from a public API with recorded parameters.

## 8. Risks

* Credit reporting dominance swamps the clusters. Mitigated by stratified sampling, and the
  comparison is reported rather than hidden.
* Templated credit repair filings form their own artificial cluster. This is a finding, not a
  failure, and gets its own section.
* Redaction runs of X dominate the vocabulary. Handled in phase 4 and verified in phase 2.
* KMeans on sparse high dimensional text is known to be weak. That is the point of phase 8.
* Deep paging limits on the API. Handled by slicing queries per product per month so no single
  query needs a large offset.

## 9. Definition of done

* `scripts/fetch_data.py` runs clean from an empty `data/raw`
* Every number in the README traceable to a CSV in `reports/`
* Baselines stated next to every headline metric
* A section on what did not work
* Pushed to GitHub with the reference repository credited
