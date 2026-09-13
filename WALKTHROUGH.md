# Walkthrough

How this project was actually built, in order, including the things that went wrong and the
decisions that turned on a measurement. The README states the results. This document explains
how they were arrived at and why each choice was made.

## The starting point

The brief was a standard unsupervised topic modelling exercise: clean text, tokenise, vectorise
with both CountVectorizer and TF IDF, cluster with KMeans at two centroid counts, interpret the
clusters, visualise with word clouds. The reference implementation was a public repository that
does exactly this on Vodafone customer tweets.

Two problems surfaced before any code was written.

**The reference dataset does not exist publicly.** The repository ships a notebook, a
requirements file and eight PNGs. The notebook mounts Google Drive and loads a private CSV.
There is nothing to download, so the work could not be reproduced from the public files at all.

**The brief's cluster names do not transfer.** It describes clusters as happy, neutral, sad and
angry. That framing suits a general tweet stream. On a complaints corpus every document is a
grievance by definition, so a sentiment axis collapses and the clusters become uninterpretable
as moods. The fix was to split the idea into two axes: themes from clustering, intensity from a
separate sentiment pass. Phase 9 later showed the intensity axis does not work either, for
reasons that turned out to be interesting.

## Choosing a dataset

Four criteria, in priority order: fetchable without an account, genuinely financial, free text
of the right length, and carrying labels that permit honest evaluation.

The Consumer Financial Protection Bureau complaint database was the only candidate satisfying
all four. Financial PhraseBank is too small and is supervised sentiment. SEC filings are far too
long and unlabelled. Reddit requires registration and has no labels. Kaggle Twitter sets need an
account and have inconsistent licensing.

The deciding factor was the fourth criterion. The bureau assigns a product and an issue to every
complaint by hand. That converts an unscoreable clustering exercise into one with real metrics,
which is the single thing the reference solution cannot do.

## Phase 1: acquisition, and two bugs

The first fetch script was written, smoke tested on 22 requests, and handed over. It died at 70
percent with HTTP 429 and saved nothing.

Two faults, both mine. It hammered the API at 1.4 requests a second and tripped the rate
limiter. Worse, it held everything in memory and only wrote parquet at the very end, so a
failure at 70 percent discarded 276 strata of work.

The rewrite checkpoints every stratum to disk the moment it arrives, using an atomic write, so
an interrupted run resumes rather than restarting. It throttles to one request a second, honours
`Retry-After`, and treats 429 as an expected condition rather than a fatal one. The resume was
verified rather than assumed: a 24 request run followed by the identical command made zero
network calls and produced identical row counts.

A third quirk cost time and is now a code comment: consumerfinance.gov returns 403 for any
custom User Agent string, including browser like ones, and accepts the stock requests agent.

## Phase 1b: the silent seven month hole

The first full pull came back 2,735 rows under its ceiling. That gap was investigated rather
than shrugged at, and it turned out to be the most instructive bug in the project.

**The bureau renamed its product categories in August 2023.** Nine categories before, eleven
after. The product list used the current names. Querying a current name against an earlier month
does not error, it **matches nothing and returns an empty result**.

So five of eleven categories silently had zero rows for January to July 2023 while the six whose
names never changed had full coverage. The corpus looked fine and was quietly malformed: one
dataset containing two incompatible labelling schemes with a seven month hole in five classes.

The fix was to start at September 2023, the first month cleanly on one scheme. The alternative,
fetching the old names and mapping them forward, was rejected: one old category splits into two
new ones, so recovering it means inferring labels that the clustering is then scored against.
Contaminating the ground truth to gain seven months of data is the wrong trade in a project
whose whole argument is honest evaluation.

Two guards were added. The script now warns loudly if any returned product label falls outside
the expected set. And checkpoint filenames now include the page size, because raising the
per stratum cap would otherwise have silently reused the smaller shards from an earlier run.

**The transferable lesson:** a filter that matches nothing returns an empty set, not an error.
Any pipeline filtering on a categorical value needs to assert that the value exists in the data,
because the failure mode is a quiet hole rather than a crash.

## Phase 2: auditing before modelling

Nothing was modelled until the corpus had been measured. The audit found four things that
shaped everything after it.

**Templated bulk filings dominate.** Credit repair firms file near identical Fair Credit
Reporting Act dispute letters in volume. Exact duplicates are 5.4 percent of the stratified
corpus and 37.6 percent of the natural one, but the near duplicate surplus is 9.9 and 55.2
percent. One template group holds 1,195 documents. The groups are paraphrases of one another,
differing only in wording, so exact matching catches under half of it. MinHash over five word
shingles was needed.

**Redaction is everywhere.** 81 percent of narratives contain runs of X masking names and
account numbers, 4.25 percent of all characters.

**Length varies systematically by product.** Mortgage complaints run 211 words at the median,
debt collection 95. This was flagged as a confound at the time, and became the spine of phases 5
to 8.

**Zero non ASCII characters.** The reference solution includes an ASCII filtering step written
for multilingual tweets. On this data it would be pure ceremony, so it was dropped and the
measurement recorded as the reason.

## Phases 3 and 4: cleaning, and two bugs the tests caught

The cleaning pipeline does what the audit called for. Writing unit tests for it surfaced two
defects that would otherwise have been invisible.

**MinHash was not reproducible.** It used Python's built in `hash()`, which is salted per
process, so the same script on the same data gave different duplicate counts between runs. In a
pipeline whose entire claim is reproducibility that is disqualifying. Replaced with blake2b, and
verified: two independent audit runs now produce byte identical output.

**Near duplicate detection ran on the wrong text.** It compared the stopword stripped version,
which halves document length and makes two paraphrases of a template look far less similar than
they are. It now compares the original narrative, which also keeps the numbers consistent with
the phase 2 audit. This surfaced because a test using four real paraphrases of an FCRA template
asserted they should collapse to one, and they did not.

The headline result of this phase was not a bug. **Collapsing templated filings drops credit
reporting from 72.9 percent to 46.7 percent of the natural corpus.** The apparent dominance of
that category is substantially an artefact of bulk filing rather than a count of affected
consumers. The majority class baseline moves with it, which matters because quoting the wrong
baseline would flatter or unfairly penalise every later result.

## Phase 5: the hypothesis becomes falsifiable

Two findings, both verified rather than inferred.

**The reference solution's `max_df=0.7` removes zero terms on this corpus.** The most widespread
term after stopword removal is "account" at 52.2 percent; nothing reaches 70. A CountVectorizer
at 1.0 and one at 0.7 produce an identical vocabulary and a byte identical matrix. The reason is
structural: the reference corpus is one company and one service, where a term like "network"
genuinely appears in most documents. This corpus spans eleven products from hundreds of firms.
A hyperparameter doing real work on one corpus does nothing on another.

**The length confound was quantified.** 74.8 percent of the pairwise Euclidean distance between
raw count vectors is explained by the difference in document length alone. Under L2 normalised
TF IDF it is 2.4 percent. Raw count row norms span 2.0 to 194.0; TF IDF norms are exactly 1.0.

This turned a vague worry into a prediction that could be wrong: if raw counts score well in
phase 6, the score cannot be trusted until the clusters are checked for being length bands.

## Phase 6: the prediction confirmed, and a bigger problem found

At K = 8 the raw count clustering explains **61.5 percent** of the variance in document length.
Its cluster medians run monotonically: 45, 106, 132, 164, 255, 486, 881, 2,238 words. Those are
not eight topics, they are eight length bands, and the last contains four documents. TF IDF on
the identical corpus with the identical algorithm and the identical K gives medians spanning 30
to 157 words with cluster sizes between 739 and 3,380.

Then the larger finding. **Silhouette rates the broken model 26 times better than the good one**,
0.196 against 0.008. The reference solution's own future work section proposes adding silhouette
analysis and the elbow method to validate the choice of K. On this corpus that advice, followed
literally, selects the model that failed.

No two internal metrics agreed on K either. Silhouette and Calinski Harabasz pick 2, Davies
Bouldin picks 20, and inertia has no visible elbow. The honest answer from internal metrics
alone is that they cannot tell us.

## Phase 7: what the labels say

Everything beats the 10.2 percent majority baseline and the permuted control. The best held out
result is 47.5 percent purity with ARI 0.245, a 4.7 fold lift over guessing from a method that
never sees a label.

With ground truth available, the phase 6 contradiction could be scored. Across the twelve
evaluated fits, **silhouette correlates with ARI at negative 0.590 and with NMI at negative
0.651**. The highest silhouette in the entire study, 0.474, has an adjusted Rand index of exactly
0.000. It is literally no better than shuffling. The internal metric was not merely unhelpful,
it was inverted.

Two clusters are nearly pure: student loan 97.8 percent, prepaid card 97.2 percent, both with
vocabulary nobody uses about anything else. Two clusters holding 47 percent of the corpus are
generic grievance language, because someone describing an unauthorised charge uses the same
words whether the instrument was a card, a checking account or a transfer. That is the honest
ceiling of the method, not a tuning failure.

A phase 6 prediction was also tested and found wrong. The statutory template cluster was
predicted to hurt purity; merging it moves purity from 0.3761 to 0.3747, so it does not. Cluster
4 is not concentrated in credit reporting at all, because consumers use the template to dispute
any account type. It is a register cluster cutting across the whole taxonomy, which is a more
interesting object than the prediction assumed. The phase 6 document was corrected in place.

## Phase 8: the mechanism, corrected

NMF and LDA were fitted on the same split with the same gates and scored with the same metrics.
The reference argues for KMeans over both without fitting either.

**The most useful result corrects an earlier conclusion of this project.** LDA consumes the
identical raw count matrix that made KMeans collapse into length bands, and is immune: 0.2
percent of length variance explained against 61.5. LDA is in fact the least length coupled
method tested, below both TF IDF methods.

So the defect was never "raw counts are bad". It was raw counts under **Euclidean distance**.
KMeans measures straight line distance in a space where vector magnitude is document length.
LDA never measures distance, it models each document as a mixture drawn from a Dirichlet, which
normalises the document away by construction. "Use TF IDF instead of counts" is a rule of thumb;
this is the mechanism.

On the reference's four claims: speed is half right, since NMF fits in 1.27 seconds against
KMeans at 1.00 and only LDA is slow. Accuracy is right by accident, tied with NMF rather than
ahead. Hard assignment is genuinely vindicated and is its strongest argument, since NMF places
over half its weight on a single topic for only 37 percent of documents at K = 11.
Interpretability goes against it: NMF is the most balanced method tested and LDA leaves one of
eleven topics empty.

KMeans and NMF score almost identically yet agree with each other at only ARI 0.358, about as
weakly as either agrees with the official taxonomy. There is no single correct partition of this
corpus, only several defensible ones.

## Phase 9: the visuals, and an honest failure

Word clouds were built from TF IDF centroid weights rather than by concatenating each cluster's
text, because concatenation lets the longest documents supply most of the tokens, which is the
length confound reappearing in the visualisation layer. Each cloud is paired with a ranked bar
chart showing what the cloud cannot: the gap between first and second place, whether a term is
characteristic or merely common, and any number at all.

The TSNE projection is shown coloured two ways. The cluster panel is tidy by construction and
proves nothing, since KMeans drew those boundaries. The label panel is the honest picture, and
the continuous smear at its centre is the 47.5 percent purity ceiling in visual form.

The sentiment layer failed. Every document is a complaint filed with a regulator, yet VADER
scores 45.1 percent of them positive. What it measures is register, not grievance. The highest
scoring complaint in the corpus, at the maximum +1.000, is a hostile legal accusation containing
"good faith" and "fair dealings". Templated filings average +0.058 against negative 0.044 for
unique ones, because credit repair boilerplate is polite. Ranking the themes by VADER is almost
exactly ranking them by formality.

VADER is a lexicon tuned on social media, where intensity comes from punctuation, capitals and
emoji. A calm chronological account of losing four thousand dollars contains no negative words
and is devastating. No threshold tuning fixes that.

## What I would do differently

**Audit the categorical filters before the first full fetch.** The taxonomy hole cost a nine
minute run and would have cost far more had it reached the modelling stage. A single
aggregation query per month at the start would have caught it.

**Write the reproducibility test before the pipeline, not after.** The salted hash bug existed
for two phases before a test found it. Any component whose output feeds a published number
should be run twice and diffed as a matter of course.

**Check the sentiment tool against a handful of documents before building a layer on it.** Five
minutes reading VADER's output on ten complaints would have revealed the register problem before
it became a planned deliverable.

**Treat the reference implementation's parameters as hypotheses.** Three of its choices,
`max_df=0.7`, ASCII filtering, and raw counts, turned out to be inert, unnecessary or actively
harmful on this corpus. None of that was visible without measuring. Copying them across would
have produced a project that looked complete and was wrong in ways nobody would have noticed.
