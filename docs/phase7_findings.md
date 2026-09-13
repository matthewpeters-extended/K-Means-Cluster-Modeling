# Phase 7 findings: evaluation against ground truth

Produced by `scripts/evaluate_clusters.py`. Tables in `reports/evaluation.csv`,
`contingency_k8_tfidf.csv` and `company_confound.csv`. Figures 11 and 12.

This is the phase the reference solution has no equivalent of. Every clustering from phase 6 is
scored against the product and issue labels the bureau assigns by hand, on the held out test
split as well as on train, with two baselines beside every number.

Baselines on this corpus:

* **Majority class: 10.2 percent.** Always guessing "checking or savings account", the largest
  class in the deduplicated stratified corpus.
* **Permuted control: 10.3 to 12.3 percent** depending on K. The same cluster sizes with
  membership shuffled. This is what purity scores when the clustering carries no information.

## E1. Held out results against product labels

<table>
<tr><th>Representation</th><th>K</th><th>Purity</th><th>Permuted</th><th>ARI</th><th>NMI</th></tr>
<tr><td>tfidf_unigram_svd100</td><td>11</td><td><b>0.475</b></td><td>0.123</td><td>0.245</td><td><b>0.397</b></td></tr>
<tr><td>tfidf_unigram</td><td>11</td><td>0.467</td><td>0.123</td><td>0.233</td><td>0.386</td></tr>
<tr><td>tfidf_bigram</td><td>11</td><td>0.464</td><td>0.123</td><td><b>0.255</b></td><td>0.389</td></tr>
<tr><td>tfidf_unigram_svd100</td><td>8</td><td>0.432</td><td>0.119</td><td>0.209</td><td>0.376</td></tr>
<tr><td>tfidf_unigram</td><td>8</td><td>0.384</td><td>0.119</td><td>0.195</td><td>0.337</td></tr>
<tr><td>tfidf_bigram</td><td>8</td><td>0.360</td><td>0.118</td><td>0.193</td><td>0.342</td></tr>
<tr><td>count_unigram</td><td>11</td><td>0.198</td><td>0.114</td><td>0.014</td><td>0.141</td></tr>
<tr><td>count_unigram</td><td>8</td><td>0.188</td><td>0.112</td><td>0.014</td><td>0.123</td></tr>
<tr><td>tfidf_unigram</td><td>2</td><td>0.153</td><td>0.107</td><td>0.039</td><td>0.119</td></tr>
<tr><td>count_unigram</td><td>2</td><td>0.114</td><td>0.105</td><td>0.000</td><td>0.007</td></tr>
</table>

**The best result is 47.5 percent purity against an 11 class taxonomy, with a majority baseline
of 10.2 percent and a permuted control of 12.3 percent.** That is a genuine 4.7 fold lift over
guessing, from a method that never sees a single label during fitting.

It is also, plainly, not close to a solved problem. Fifty two percent of documents sit in a
cluster whose majority product is not their own. An ARI of 0.245 means the partition agrees
with the official taxonomy considerably better than chance and nowhere near completely.

## E2. The internal metrics were not just unhelpful, they were inverted

Phase 6 C1 showed silhouette preferring the raw count model. With ground truth available, that
preference can now be scored. Sorting all twelve evaluated fits by silhouette, best first:

<table>
<tr><th>Representation</th><th>K</th><th>Silhouette</th><th>Length variance explained</th><th>ARI (test)</th></tr>
<tr><td>count_unigram</td><td>2</td><td>0.474</td><td>42.6%</td><td>0.000</td></tr>
<tr><td>count_unigram</td><td>8</td><td>0.196</td><td>61.5%</td><td>0.014</td></tr>
<tr><td>count_unigram</td><td>11</td><td>0.186</td><td>61.2%</td><td>0.014</td></tr>
<tr><td>tfidf_unigram_svd100</td><td>2</td><td>0.065</td><td>0.0%</td><td>0.039</td></tr>
<tr><td>tfidf_unigram_svd100</td><td>8</td><td>0.037</td><td>2.4%</td><td>0.209</td></tr>
<tr><td>tfidf_unigram_svd100</td><td>11</td><td>0.037</td><td>12.8%</td><td>0.245</td></tr>
<tr><td>tfidf_unigram</td><td>11</td><td>0.008</td><td>12.8%</td><td>0.233</td></tr>
<tr><td>tfidf_bigram</td><td>11</td><td>0.004</td><td>16.0%</td><td>0.255</td></tr>
</table>

Across the twelve fits:

* silhouette against ARI: **(0.590)**
* silhouette against NMI: **(0.651)**
* length variance explained against ARI: **(0.440)**

Negative numbers are shown in parentheses.

**The correlation between the internal metric and the truth is not merely weak, it is
substantially negative.** The clustering with the highest silhouette in the entire study,
count_unigram at K = 2 with 0.474, has an adjusted Rand index of exactly 0.000. It is literally
no better than shuffling.

Selecting a model by silhouette on this corpus would not have been a slightly suboptimal
choice. It would have been close to the worst possible choice, and it is precisely the
improvement the reference solution's own future work section recommends.

The reason is visible in the middle column. Silhouette rewards compact, well separated clusters
in the space you give it. Raw count space separates documents by length, and length separates
very cleanly indeed, because it is a single strong dimension. Topic structure in a sparse term
matrix is diffuse and high dimensional, so genuinely meaningful clusters look poor by
silhouette while meaningless ones look excellent.

## E3. Nothing overfits

Adjusted Rand index, train against test, for every representation and K:

<table>
<tr><th>Representation</th><th>K</th><th>Train ARI</th><th>Test ARI</th><th>Change</th></tr>
<tr><td>tfidf_unigram_svd100</td><td>11</td><td>0.240</td><td>0.245</td><td>+0.005</td></tr>
<tr><td>tfidf_bigram</td><td>11</td><td>0.243</td><td>0.255</td><td>+0.012</td></tr>
<tr><td>tfidf_unigram</td><td>11</td><td>0.225</td><td>0.233</td><td>+0.008</td></tr>
<tr><td>tfidf_unigram</td><td>8</td><td>0.191</td><td>0.195</td><td>+0.004</td></tr>
<tr><td>count_unigram</td><td>8</td><td>0.015</td><td>0.014</td><td>(0.001)</td></tr>
</table>

Test scores are marginally **higher** than train in most cases. There is no generalisation gap
at all, which is what one expects from an unsupervised method with K parameters against 14,332
documents. The vectoriser and the centroids were fit on train alone and applied unchanged to
3,583 unseen documents.

This is worth stating explicitly because the reference solution fits on the entire corpus and
therefore cannot make this claim. The claim here is modest, since there was little risk of
overfitting to begin with, but it is now evidenced rather than assumed.

## E4. Two clusters are nearly pure, and the rest are mixtures

From the contingency table, tfidf_unigram at K = 8, row percentages:

<table>
<tr><th>Cluster</th><th>Documents</th><th>Dominant product</th><th>Share</th></tr>
<tr><td>5</td><td>1,010</td><td>Student loan</td><td>97.8%</td></tr>
<tr><td>7</td><td>739</td><td>Prepaid card</td><td>97.2%</td></tr>
<tr><td>0</td><td>868</td><td>Debt collection</td><td>54.3%</td></tr>
<tr><td>3</td><td>1,036</td><td>Credit reporting</td><td>38.2%</td></tr>
<tr><td>6</td><td>3,289</td><td>Checking or savings</td><td>33.3%</td></tr>
<tr><td>1</td><td>2,990</td><td>Mortgage</td><td>33.7%</td></tr>
<tr><td>4</td><td>1,020</td><td>Vehicle loan</td><td>18.3%</td></tr>
<tr><td>2</td><td>3,380</td><td>Credit card</td><td>14.1%</td></tr>
</table>

Student loans and prepaid cards are recovered almost perfectly. Both have distinctive,
non transferable vocabulary: forbearance, MOHELA, repayment plan, school on one side; gift
card, reload, activate, balance on the other. Nobody uses those words about a mortgage.

The weak clusters are the ones whose vocabulary is shared. Clusters 2 and 6 together hold 47
percent of the corpus and are dominated by generic financial grievance language: money, bank,
account, charged, told, called. A person describing an unauthorised charge uses much the same
words whether the instrument was a credit card, a checking account or a money transfer. The
product label distinguishes them; the narrative often does not.

**This is the honest ceiling of the method.** It is not a tuning failure. Unsupervised
clustering of complaint text can only recover distinctions that the text actually encodes, and
the bureau's product taxonomy encodes distinctions that consumers frequently do not make in
their own words.

## E5. A prediction from phase 6 that turned out wrong

Phase 6 C5 identified clusters 3 and 4 as the same subject split by register: cluster 3 people
describing credit report errors in their own words, cluster 4 the Fair Credit Reporting Act
template that phase 2 catalogued as defect D2. It predicted that purity would penalise this
split, since both clusters ought to be mostly credit reporting.

Tested directly by merging the two clusters and rescoring:

* purity with clusters 3 and 4 separate: **0.3761**
* purity with clusters 3 and 4 merged: **0.3747**

Merging makes purity very slightly **worse**, not better. The prediction was wrong, and the
reason is interesting. Cluster 4 is not concentrated in credit reporting at all. Its product
mix is vehicle loan 18 percent, credit reporting 16 percent, debt collection 14 percent,
spread across nearly every category.

The statutory template is used by consumers to dispute **any** account they want removed, not
only credit report entries. So cluster 4 is a genuine register cluster that cuts across the
entire product taxonomy. That makes it a more interesting object than first assumed: the
clustering has isolated a document style, produced largely by the credit repair industry, which
is orthogonal to what the complaint is actually about.

## E6. Open question: how much of this is company name recognition?

Phase 6 flagged that mohela, amex, carvana and chase appear as distinctive cluster terms.
Measured at K = 8 on train:

<table>
<tr><th>Representation</th><th>Company names as share of distinctive terms</th><th>NMI with product</th><th>NMI with company</th></tr>
<tr><td>count_unigram</td><td>28.1%</td><td>0.123</td><td>0.088</td></tr>
<tr><td>tfidf_unigram</td><td>26.0%</td><td>0.335</td><td>0.244</td></tr>
<tr><td>tfidf_bigram</td><td>27.1%</td><td>0.339</td><td>0.263</td></tr>
<tr><td>tfidf_unigram_svd100</td><td>30.2%</td><td>0.369</td><td>0.262</td></tr>
</table>

Roughly **one in four distinctive terms is a company name**, and mutual information with the
company is about 71 to 78 percent of mutual information with the product.

**The necessary control:** product and company are already coupled in the data, at
NMI(product, company) = 0.425 across 1,064 companies and 11 products. A firm that only services
mortgages generates only mortgage complaints. So a clustering that recovers products perfectly
would also show high mutual information with company, and these numbers cannot separate
"learned the topic" from "learned the firm".

What can be said honestly: a substantial minority of the signal the clustering keys on is
proper nouns rather than complaint semantics. Whether that counts as cheating depends on the
use case. For routing a complaint to the right department it is perfectly useful. For
discovering what consumers are unhappy about, it is a shortcut that would fail on any firm not
present in the training data.

**This belongs in the README's "what did not work" section**, and the clean follow up is to
strip company name tokens from the vocabulary and rerun, which is recorded as future work
rather than claimed as done.

## What this means overall

* The method works, modestly and measurably: 47.5 percent purity against 10.2 percent majority
  and 12.3 percent permuted, ARI 0.245, on held out data.
* K = 11 beats K = 8 beats K = 2 on every metric and every representation. More granularity
  recovers more of the taxonomy, with no sign of the curve turning over by K = 11.
* TF IDF beats raw counts by a factor of roughly 14 on ARI. The representation mattered far
  more than the choice of K, the algorithm, or anything else tested.
* SVD to 100 components is the best single configuration, and the cheapest.
* Internal validation metrics were actively misleading on this corpus, which is the single most
  transferable lesson in the project.
