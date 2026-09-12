# Sources

## Reference solution

Carlos Rodriguez, `NLP-KMeans-Topic-Modeling`.
https://github.com/carlosrod723/NLP-KMeans-Topic-Modeling

Unsupervised topic modelling of Vodafone customer tweets. CountVectorizer with min_df and
max_df gates feeding KMeans at K = 2 and K = 6, with word clouds masked to the Twitter logo.
MIT licensed. Declared stack: numpy, pandas, scikit learn, nltk, matplotlib, seaborn,
wordcloud, regex, jupyter.

This project takes that method as its starting specification and ports it to a finance corpus.
No code is copied. The evaluation gaps in that repository are listed in `PLAN.md` section 7 and
are the main thing this project adds.

Important practical note. The tweet corpus that repository uses is not in the repository. The
notebook loads it from the author's own Google Drive, so it cannot be reproduced from the
public files. That is why a different, openly fetchable dataset was chosen.

## Chosen dataset

**Consumer Financial Protection Bureau, Consumer Complaint Database.**

* Landing page: https://www.consumerfinance.gov/data-research/consumer-complaints/
* API documentation: https://cfpb.github.io/api/ccdb/
* Query endpoint used:
  `https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/`
* Licence: United States federal government work, public domain. No key, no account, no quota.

### Why this one

* It is genuinely fintech. Credit cards, checking accounts, mortgages, debt collection, money
  transfer and virtual currency, payday lending, prepaid cards.
* The free text field is exactly the kind of messy, informal, user written prose the method
  needs, so the cleaning work in the brief is real work and not decoration.
* It carries hand assigned labels, which converts an unscoreable clustering exercise into one
  with real metrics.
* It is fetchable by script, so the whole project reproduces from an empty folder.
* It is large enough that sampling strategy becomes a genuine decision.

### Fields used

* `complaint_what_happened`, the narrative. This is the corpus.
* `product` and `sub_product`, the coarse label. Evaluation ground truth.
* `issue` and `sub_issue`, the fine label. Secondary evaluation ground truth.
* `company`, `state`, `date_received`, `submitted_via`, `company_response`, `timely`.
  Segmentation and exploratory analysis only, never fed to the clustering.

### Known characteristics, verified live on 2026 09 12

Complaints received in calendar 2024 that carry a narrative: 815,822.

Product breakdown of those:

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

So credit reporting is roughly 77 percent of the narratives. This is the dominant modelling
constraint and is addressed by stratified sampling in `PLAN.md` section 4.

API behaviour confirmed by direct probing:

* `size` accepts at least 5,000 records per request
* `frm` offset paging works
* `has_narrative=true` filters to rows with usable text
* `no_aggs=true` skips the aggregation block and returns faster
* Passing `format=json` returns a 404, the JSON body is the default, so do not pass it

### Privacy and ethics

Narratives are published by the bureau only where the consumer gave consent. Personal details
are already redacted by the bureau as runs of the letter X before publication. Company names
are public. No attempt is made in this project to reidentify anyone, and no complaint is quoted
in full in the README.

## Alternatives considered and rejected

* **Financial PhraseBank.** Sentiment labelled finance sentences. Rejected because it is only
  about 5,000 short sentences, too small and too clean for a clustering and text cleaning
  exercise, and it is supervised sentiment rather than topic discovery.
* **SEC EDGAR filings, risk factor sections.** Genuinely interesting finance text and freely
  available. Rejected because documents run to tens of thousands of words each, which makes it
  a document summarisation problem rather than the short text clustering the brief describes,
  and there are no topic labels to score against.
* **Reddit finance and investing subreddits.** Closest in spirit to the original tweet corpus.
  Rejected because API access now requires registration and is rate limited, the content is not
  redacted so it carries personal information risk, and there are no labels.
* **Twitter or X financial sentiment sets on Kaggle.** Rejected because most require a Kaggle
  account to download, licensing is inconsistent, and many are redistributions that breach the
  platform terms. Reproducibility was the deciding factor.
* **Bank customer review sets such as Trustpilot scrapes.** Rejected for terms of service and
  provenance reasons.

The deciding criteria were, in order: fetchable without an account, genuinely financial, free
text of the right length, and carrying labels that permit honest evaluation. Only the bureau
database satisfies all four.
