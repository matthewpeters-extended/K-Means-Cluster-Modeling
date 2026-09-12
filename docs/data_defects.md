# Data defects

Defects found in the source data, and what was done about each. Written as they are found,
before modelling, so that nothing gets explained away after the fact.

## D1. The product taxonomy changed in August 2023

**Severity: high. Found before any modelling. Fixed by narrowing the window.**

The bureau renamed and resplit its product categories partway through 2023. The first corpus
pull covered January 2023 to December 2024 using the current category names, and came back
2,735 rows short of its ceiling with no error raised.

Cause. Querying `product=Credit card` against a month that predates the rename matches nothing
and returns an empty result set rather than an error. Five of the eleven categories were
therefore silently absent for the first seven months of the window, while the six categories
whose names did not change had full coverage. The corpus looked fine and was quietly malformed:
a single corpus containing two different labelling schemes, with a seven month hole in five of
its eleven classes.

Evidence, from the live aggregation endpoint:

<table>
<tr><th>Month</th><th>New name present</th><th>Old name present</th><th>Distinct products</th></tr>
<tr><td>2023 05</td><td>0</td><td>30,318</td><td>9</td></tr>
<tr><td>2023 06</td><td>0</td><td>30,513</td><td>9</td></tr>
<tr><td>2023 07</td><td>0</td><td>29,458</td><td>9</td></tr>
<tr><td>2023 08</td><td>7,125</td><td>26,765</td><td>14</td></tr>
<tr><td>2023 09</td><td>30,785</td><td>0</td><td>11</td></tr>
</table>

August 2023 is a transition month carrying both schemes at once.

The mapping:

* Credit reporting, credit repair services, or other personal consumer reports becomes
  Credit reporting or other personal consumer reports
* Credit card or prepaid card splits into two separate categories, Credit card and Prepaid card
* Payday loan, title loan, or personal loan gains "or advance loan"
* Debt or credit management is new and has no predecessor
* Debt collection, Checking or savings account, Mortgage, Student loan, Vehicle loan or lease,
  and Money transfer, virtual currency, or money service are unchanged

**Decision. Start the corpus at September 2023, the first month that is cleanly on one scheme.**

The alternative was to fetch the older names too and map them forward. That was rejected. The
product label is our evaluation ground truth, and one of the old categories, Credit card or
prepaid card, maps to two new categories rather than one. Recovering that split would mean
inferring the boundary from the sub product field. Any error in that inference lands directly
in the labels we score the clustering against, which means we would be measuring our own
mapping as much as the clustering. Trading seven months of extra data for contaminated ground
truth is a bad trade in a project whose entire argument is honest evaluation.

Cost of the decision: the window narrows from 24 months to 16, September 2023 to December 2024.
The per stratum cap rises from 75 to 115 to hold the corpus near 20,000 documents.

**Guards added so this cannot recur silently.**

* The fetch script now compares every returned product label against the expected set and warns
  loudly if anything unrecognised appears. A future rename surfaces immediately.
* The taxonomy change, the cutover month and the superseded names are documented in a comment
  directly above the product list in `scripts/fetch_data.py`.
* Checkpoint filenames now include the requested page size. Previously, raising the per stratum
  cap would silently reuse the smaller checkpoints from an earlier run and cap the corpus
  without saying so. That is a defect of the same family as this one: a size change that fails
  quietly rather than loudly.

**Wider lesson, worth stating in the README.** A filter that matches nothing returns an empty
set, not an error. Any pipeline that filters by a categorical value needs to assert that the
value actually exists in the data, because the failure mode is a quiet hole rather than a
crash. This one was caught only because a row count came in below its ceiling and the shortfall
was investigated instead of shrugged at.

## D2. Class imbalance in the natural distribution sample

**Severity: expected, not a defect as such. Handled by design.**

The natural sample reflects how complaints actually arrive, and credit reporting dominates it.
Measured on the first pull, credit reporting was slightly over half of the natural sample even
before the old taxonomy rows were folded in, against roughly one eleventh in the stratified
sample. This is the reason two corpora are built rather than one, and the comparison between
them is a planned finding rather than a problem to be hidden.

## D3. Redaction artefacts

**Severity: high for modelling. Not yet addressed. Phase 4.**

The bureau masks names, dates, account numbers and amounts as runs of the letter X before
publication. These runs are expected to be among the highest frequency tokens in the corpus and
will dominate every cluster if they survive preprocessing. To be quantified in phase 2 and
stripped in phase 4.

## D4. Templated filings

**Severity: medium. Not yet addressed. Phase 4.**

Credit repair firms file large volumes of near identical narratives on behalf of clients.
Deduplicating on exact string match will not catch these because small details differ. Expected
to form an artificial cluster if left alone. Near duplicate detection over hashed token
shingles is planned for phase 4.
