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

## D2. Templated bulk filings, the largest defect in the corpus

**Severity: critical for the natural sample. Measured in phase 2. To be handled in phase 4.**

Credit repair firms file large volumes of near identical narratives on behalf of clients. The
scale of this is far worse than expected, and it is concentrated in exactly the category that
dominates the natural sample.

<table>
<tr><th>Measure</th><th>Stratified</th><th>Natural</th></tr>
<tr><td>Documents</td><td>19,979</td><td>20,000</td></tr>
<tr><td>Exact duplicates after normalising</td><td>1,085 (5.4%)</td><td>7,513 (37.6%)</td></tr>
<tr><td>Near duplicate groups</td><td>723</td><td>2,372</td></tr>
<tr><td>Surplus copies beyond one per group</td><td>1,951 (9.8%)</td><td>10,997 (55.0%)</td></tr>
<tr><td>Largest single template group</td><td>110 documents</td><td>1,167 documents</td></tr>
</table>

Measured with MinHash over five word shingles, 32 permutations banded eight by four, which
catches pairs above roughly 0.6 Jaccard similarity.

**More than half of the natural sample is redundant.** One template alone accounts for 1,167
documents, nearly six percent of that corpus. Left in place, KMeans would build centroids
around boilerplate phrasing rather than around complaint themes, and the resulting clusters
would measure which credit repair firm filed the paperwork rather than what consumers are
actually complaining about.

The templates are recognisable Fair Credit Reporting Act dispute letters. The largest group in
the stratified corpus, 110 documents, opens "In accordance with the Fair Credit Reporting act.
The List of accounts below has violated my federally protect". Several distinct groups are
paraphrases of one another, which is why exact deduplication alone is insufficient: groups of
45 and 38 documents differ only in "I understand the significance of eliminating any erroneous
accounts" against "I understand the importance of removing any incorrect accounts".

The reference solution deduplicates on exact string match only. On this corpus that would
remove 5.4 percent of the stratified sample and leave the remaining 9.8 percent of near
duplicates in place, which is the part that actually distorts the centroids.

**Decision. Deduplicate exactly, then collapse each near duplicate group to a single
representative, and keep the group sizes as a separate column.** The volume of a template is
real information about consumer behaviour and belongs in the exploratory analysis. It just must
not be allowed to vote repeatedly in the clustering.

## D3. Redaction padding

**Severity: high for modelling. Measured in phase 2. To be stripped in phase 4.**

The bureau masks names, dates, account numbers and amounts as runs of the letter X before
publication.

* 81.2 percent of stratified narratives contain at least one redaction run
* Median of 6 runs per affected document
* Redaction accounts for 4.25 percent of all characters in the stratified corpus and 6.38
  percent in the natural corpus

Untouched, these runs become among the highest frequency tokens in the vocabulary and appear in
every cluster, contributing nothing that separates one theme from another. Stripped in phase 4
before tokenisation.

## D4. Vocabulary sparsity

**Severity: medium. Measured in phase 2. Handled by frequency gates in phase 5.**

The stratified corpus yields 24,218 distinct tokens before stopword removal, and 37.5 percent
of them appear exactly once. Those single occurrence tokens are overwhelmingly typographical
errors, fragments of redacted strings and rare proper nouns. They add dimensions to the
document term matrix without adding any signal that could separate clusters. This is what the
min_df gate in phase 5 exists to remove, and the measurement gives us a basis for choosing the
threshold rather than copying one.

## D5. Narrative length varies systematically by product

**Severity: medium, and a genuine confound. Measured in phase 2.**

Median narrative length is not constant across the categories we are going to evaluate against:

<table>
<tr><th>Product</th><th>Median words</th></tr>
<tr><td>Mortgage</td><td>211</td></tr>
<tr><td>Student loan</td><td>171</td></tr>
<tr><td>Money transfer, virtual currency, or money service</td><td>168</td></tr>
<tr><td>Checking or savings account</td><td>166</td></tr>
<tr><td>Vehicle loan or lease</td><td>163</td></tr>
<tr><td>Payday loan, title loan, personal loan, or advance loan</td><td>142</td></tr>
<tr><td>Credit card</td><td>138</td></tr>
<tr><td>Prepaid card</td><td>128</td></tr>
<tr><td>Debt or credit management</td><td>104</td></tr>
<tr><td>Credit reporting or other personal consumer reports</td><td>98</td></tr>
<tr><td>Debt collection</td><td>95</td></tr>
</table>

A mortgage complaint is more than twice the length of a debt collection complaint. Raw
CountVectorizer counts scale with document length, so under Euclidean distance KMeans can
partly separate these categories on length alone rather than on vocabulary. That would inflate
the evaluation scores for reasons that have nothing to do with topic discovery.

This is a concrete argument for comparing TF IDF against raw counts rather than assuming, since
TF IDF with L2 normalisation removes the length effect. The brief asks for both representations
anyway. Phase 5 now has a specific hypothesis to test rather than a box to tick.

## D6. Class imbalance

**Severity: expected. Handled by design, and now quantified.**

<table>
<tr><th></th><th>Stratified</th><th>Natural</th></tr>
<tr><td>Largest class share</td><td>9.2%</td><td>72.9%</td></tr>
<tr><td>Ratio, largest class to smallest</td><td>1.17 to 1</td><td>324 to 1</td></tr>
</table>

The majority class baseline is therefore 9.2 percent on the stratified corpus and 72.9 percent
on the natural one. Both numbers get stated next to any clustering result computed on that
corpus. A purity score of 70 percent on the natural sample would be worse than guessing
"credit reporting" every time, and that is precisely the kind of result that gets reported as a
success in projects that skip baselines.

Only one product fell short of its per stratum ceiling: Debt or credit management returned
1,579 of a possible 1,840 because the category does not receive 115 narratives in every month.
That is a real volume limit, not a fetch defect.

## D7. Encoding, a non issue

**Severity: none. Checked and dismissed.**

Zero narratives in either corpus contain non ASCII characters. The bureau normalises text
before publication. The reference solution includes an ASCII filtering step, written for
multilingual tweets, which would be pure ceremony here. It is omitted, and this measurement is
the reason. Steps carried over from a reference implementation should be justified against your
own data, not inherited on faith.
