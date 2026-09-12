# Phase 6 findings: clustering

Produced by `scripts/run_clustering.py`. Tables in `reports/kmeans_sweep.csv`,
`clusters_k2.csv` and `clusters_k8.csv`. Figures 08, 09 and 10.

Four representations, K swept from 2 to 20, fit on the 14,332 document training split only.
76 fits in about 100 seconds.

## C1. The internal validation metric recommends the wrong answer

**The most important result in this project so far.**

Silhouette scores at K = 8:

<table>
<tr><th>Representation</th><th>Silhouette</th><th>Largest cluster</th><th>Smallest cluster</th><th>Size ratio</th><th>Normalised entropy</th></tr>
<tr><td>count_unigram</td><td>0.196</td><td>68.9%</td><td>4 documents</td><td>2,468 to 1</td><td>0.504</td></tr>
<tr><td>tfidf_unigram</td><td>0.008</td><td>23.6%</td><td>5.2%</td><td>4.6 to 1</td><td>0.910</td></tr>
<tr><td>tfidf_bigram</td><td>0.004</td><td>28.9%</td><td>2.5%</td><td>11.6 to 1</td><td>0.880</td></tr>
<tr><td>tfidf_unigram_svd100</td><td>0.037</td><td>23.8%</td><td>5.1%</td><td>4.7 to 1</td><td>0.910</td></tr>
</table>

By silhouette, raw counts are twenty six times better than TF IDF. By every other reading of
the same clustering, raw counts have failed completely: one cluster holds 68.9 percent of the
corpus, another holds four documents, and the largest is 2,468 times the size of the smallest.

The reference solution's own future work section proposes adding silhouette analysis and the
elbow method to validate the choice of K. **On this corpus that advice, followed literally,
selects the broken model.** Silhouette measures how compact and separated clusters are in the
given space. It has no way to know that the space itself encodes the wrong thing.

This is the strongest argument available for why phase 7 exists. Internal metrics score the
geometry. Only labelled ground truth can score whether the geometry means anything.

## C2. The length hypothesis is confirmed, decisively

Phase 5 V2 predicted that raw counts would partition by document length. The test was eta
squared: the share of variance in document length explained by cluster membership.

<table>
<tr><th>K</th><th>count_unigram</th><th>tfidf_unigram</th><th>tfidf_bigram</th><th>tfidf_unigram_svd100</th></tr>
<tr><td>2</td><td>42.6%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td></tr>
<tr><td>8</td><td>61.5%</td><td>13.9%</td><td>7.3%</td><td>2.4%</td></tr>
<tr><td>20</td><td>69.1%</td><td>15.3%</td><td>13.2%</td><td>16.6%</td></tr>
</table>

At K = 8 the raw count clustering explains 61.5 percent of the variance in document length.
The cluster medians, sorted, tell the story without any statistics at all:

<table>
<tr><th>Cluster</th><th>Documents</th><th>Median words</th></tr>
<tr><td>1</td><td>9,870</td><td>45</td></tr>
<tr><td>6</td><td>796</td><td>106</td></tr>
<tr><td>3</td><td>1,743</td><td>132</td></tr>
<tr><td>0</td><td>1,433</td><td>164</td></tr>
<tr><td>7</td><td>308</td><td>255</td></tr>
<tr><td>5</td><td>160</td><td>486</td></tr>
<tr><td>2</td><td>18</td><td>881</td></tr>
<tr><td>4</td><td>4</td><td>2,238</td></tr>
</table>

Monotonic. These are not eight topics, they are eight length bands, and cluster 4 is simply
"the four longest complaints in the corpus". Figure 10 shows it directly.

The same algorithm on L2 normalised TF IDF gives cluster medians spanning 30 to 157 words with
sizes between 739 and 3,380. The difference is entirely the representation. The clustering
algorithm, the corpus and the value of K are identical.

## C3. No internal metric agrees with any other on the best K

Best K by each metric:

<table>
<tr><th>Representation</th><th>Silhouette</th><th>Davies Bouldin</th><th>Calinski Harabasz</th></tr>
<tr><td>count_unigram</td><td>2</td><td>12</td><td>2</td></tr>
<tr><td>tfidf_unigram</td><td>2</td><td>20</td><td>2</td></tr>
<tr><td>tfidf_bigram</td><td>20</td><td>20</td><td>2</td></tr>
<tr><td>tfidf_unigram_svd100</td><td>2</td><td>20</td><td>2</td></tr>
</table>

Silhouette and Calinski Harabasz almost always pick K = 2, the smallest value offered. Davies
Bouldin almost always picks K = 20, the largest. They are pointing at opposite ends of the
range. Neither is identifying structure; both are responding to the monotone tendencies of
their own formulas on sparse high dimensional data, where every point is far from every other
point and silhouette values sit near zero for any sensible partition.

The elbow plot is no better. Inertia declines smoothly with no visible knee for any TF IDF
representation.

**So the honest answer to "what is the optimal K" from internal metrics alone is: they cannot
tell us.** That is a real finding, not a failure to find one. It is also exactly why the brief's
instruction to compare two chosen values of K by interpretation was not unreasonable, and why
the evaluation against product labels in phase 7 is the part that will actually decide.

## C4. K = 2 is coherent and all three TF IDF variants agree

At K = 2, TF IDF splits the corpus into credit reporting and debt on one side, and money,
cards and banking on the other:

<table>
<tr><th>Representation</th><th>Split</th><th>Distinctive terms of the smaller cluster</th></tr>
<tr><td>tfidf_unigram</td><td>20.7 / 79.3</td><td>report, credit, debt, reporting, consumer, collection, information</td></tr>
<tr><td>tfidf_bigram</td><td>19.0 / 81.0</td><td>report, credit report, credit, reporting, debt, consumer, collection</td></tr>
<tr><td>tfidf_unigram_svd100</td><td>20.5 / 79.5</td><td>report, credit, debt, reporting, consumer, collection, information</td></tr>
</table>

Three different feature spaces, built independently, converge on the same 20 to 80 split with
the same semantics. That consistency is meaningful evidence that the boundary is real rather
than an artefact of one representation.

Raw counts at K = 2 produce an 89 to 11 split whose distinctive terms are gift, vanilla,
lexington, visa, google, carvana, cashapp. Those are brand names appearing in long documents.
It is not a theme.

## C5. K = 8 on TF IDF yields interpretable, business aligned themes

<table>
<tr><th>Cluster</th><th>Share</th><th>Distinctive terms</th><th>Reading</th></tr>
<tr><td>2</td><td>23.6%</td><td>car, vehicle, charged, repair, title, dealership, rate, sale</td><td>Vehicle finance and dealer disputes</td></tr>
<tr><td>6</td><td>22.9%</td><td>bank, money, account, fund, transaction, transfer, app, check, deposit</td><td>Banking, transfers and funds access</td></tr>
<tr><td>1</td><td>20.9%</td><td>payment, mortgage, loan, pay, paid, month, interest, late, fee, escrow</td><td>Loan and mortgage servicing</td></tr>
<tr><td>3</td><td>7.2%</td><td>report, credit, inquiry, remove, reporting, inaccurate, identity, theft</td><td>Credit report accuracy disputes</td></tr>
<tr><td>4</td><td>7.1%</td><td>consumer, act, reporting, matter, violation, financial, usc, fair, rights</td><td>Statutory boilerplate, the templated filings</td></tr>
<tr><td>5</td><td>7.0%</td><td>mohela, loan, student, forbearance, repayment, application, school</td><td>Student loan servicing</td></tr>
<tr><td>0</td><td>6.1%</td><td>debt, collection, collect, collector, validation, creditor, agency</td><td>Debt collection practices</td></tr>
<tr><td>7</td><td>5.2%</td><td>card, gift, amex, declined, express, purchase, store</td><td>Card declines and gift cards</td></tr>
</table>

Two observations worth carrying into phase 7.

**Clusters 3 and 4 are the same subject matter split by register.** Cluster 3 is a person
describing an inaccurate credit report in their own words. Cluster 4 is the Fair Credit
Reporting Act template that phase 2 identified as D2, recognisable by usc, act, violation and
rights. The clustering has separated the credit repair industry's paperwork from genuine
individual complaints, which is a more useful distinction than the product label offers, and
one the product label cannot express. It is also a warning for phase 7: purity against
`product` will penalise this split even though it is arguably the better one.

**Several clusters key on company names.** mohela, amex, carvana, lexington and chase all
appear as distinctive terms. Company name is a strong predictor of product, so the clustering
is partly learning a firm to product mapping rather than complaint semantics. This is worth
quantifying in phase 7 and is a candidate entry for a "what did not work" section.

## C6. SVD compresses aggressively and loses little

100 components retain 21.3 percent of the variance of the 6,885 dimensional TF IDF matrix, yet
produce clusters nearly identical to the full space at K = 2 (20.5 versus 20.7 percent) and
K = 8 (size ratio 4.7 versus 4.6, same themes). It also has the lowest length coupling of any
representation at K = 8, 2.4 percent.

The practical read is that most of the variance in a sparse term matrix is noise, and the
clustering structure survives heavy compression. The efficiency argument is real too: 19 fits
in 6.3 seconds against 17.0 for the full matrix.

## C7. Bigrams cost a lot and add nothing visible

tfidf_bigram has 46,531 features against 6,885, takes 60.5 seconds to sweep against 17.0, and
produces a less balanced partition at K = 8 (size ratio 11.6 against 4.6). Its K = 2 split is
the same as the unigram one. On the evidence so far the extra dimensionality is not earning its
cost, though phase 7 gets the final word.

## What phase 7 inherits

* Fitted train and test labels for K = 2 and K = 8 across all four representations, saved to
  `data/processed/features/labels_*.npz`, so nothing needs refitting
* A clear prediction to test: raw counts should score poorly against product labels despite
  their excellent silhouette, and if they do not, the reason needs investigating
* Two specific questions: how much of the clustering is company name recognition, and how
  should the register split between clusters 3 and 4 be scored when the label scheme cannot
  represent it
