# Phase 5 findings: vectorisation

Built by `scripts/build_features.py`. Figure `reports/figures/07_vectorisation.png`, tables in
`reports/vectoriser_gate_sweep.csv`, `vectoriser_comparison.csv` and `length_coupling.csv`.

The corpus was split first: 17,915 documents into 14,332 train and 3,583 test, stratified by
product, seed 42. Every vectoriser is fit on the training split alone and applied to test.
The reference solution fits on the entire corpus, which makes generalisation unmeasurable.

## V1. The reference solution's max_df gate removes nothing on this corpus

**The single clearest result in this phase.**

The reference uses `max_df=0.7`, filtering terms that appear in more than 70 percent of
documents. On this corpus that threshold removes **zero terms**. The most widespread term after
stopword removal is "account", present in 52.2 percent of documents. Nothing reaches 70.

Verified directly rather than inferred: a CountVectorizer at `max_df=1.0` and one at
`max_df=0.7` produce an identical vocabulary and a byte identical matrix, 14,332 by 6,885 in
both cases.

<table>
<tr><th>max_df</th><th>Terms removed</th><th>Which</th></tr>
<tr><td>1.0</td><td>0</td><td></td></tr>
<tr><td>0.9</td><td>0</td><td></td></tr>
<tr><td>0.7</td><td>0</td><td>the reference gate, inert</td></tr>
<tr><td>0.5</td><td>1</td><td>account</td></tr>
<tr><td>0.3</td><td>6</td><td>account, payment, credit, time, would, received</td></tr>
<tr><td>0.2</td><td>31</td><td>the above plus information, loan, year, told, day, company</td></tr>
</table>

The reason is structural, not accidental. The reference corpus is Vodafone complaint tweets, a
single company and a single service, where a term like "vodafone" or "network" genuinely does
appear in most documents. This corpus spans eleven financial products from hundreds of firms,
so no single term is that pervasive. **A hyperparameter that was doing real work on one corpus
does nothing at all on another.**

`max_df` is therefore kept at 1.0 by default and the value is not copied across for
appearances. The finding itself is more useful than the parameter would have been.

One loose end from phase 3 is resolved here. F4 noted that "would" survives cleaning at rank 7
because it is not an NLTK stopword, and asked whether `max_df` would catch it. It would not, at
any sane threshold: "would" appears in 33.3 percent of documents, so removing it needs
`max_df=0.3`, which would also discard "credit" and "payment". It stays in the vocabulary.

## V2. Raw counts encode document length. TF IDF does not.

Phase 3 finding F2 predicted that raw counts would let KMeans separate categories on length
rather than content, because median length runs from 211 words (mortgage) to 95 (debt
collection). That prediction is now measured and confirmed.

<table>
<tr><th>Representation</th><th>Row norm correlation with length</th><th>Pairwise distance correlation with length gap</th><th>Variance explained by length alone</th></tr>
<tr><td>count_unigram</td><td>0.953</td><td>0.865</td><td>74.8%</td></tr>
<tr><td>tfidf_unigram</td><td>0.000</td><td>(0.155)</td><td>2.4%</td></tr>
<tr><td>tfidf_bigram</td><td>0.000</td><td>(0.161)</td><td>2.6%</td></tr>
</table>

Negative numbers are shown in parentheses.

**Three quarters of the pairwise Euclidean distance between raw count vectors is explained by
nothing more than the difference in document length.** KMeans minimises Euclidean distance to
centroids, so a KMeans fit on raw counts is substantially a length based partition wearing the
costume of topic discovery.

The row norms make the mechanism plain. Under raw counts a document vector has norm between
2.0 and 194.0, median 10.8. Under L2 normalised TF IDF every document has norm exactly 1.0, so
length carries no weight at all and only the direction of the vector, its mix of terms, can
separate documents.

This does not mean raw counts will score badly in phase 6. It means that **if they score well,
the score cannot be trusted without checking whether the clusters are length bands.** Phase 7
now has a specific test to run: measure whether cluster assignment correlates with document
length, separately for each representation. A high purity score from raw counts alongside a
strong length correlation is a spurious result, and would be reported as one.

## V3. The min_df gate is cheap in vocabulary and expensive in tokens

Measured on the training split, at 19,001 distinct terms before gating:

<table>
<tr><th>min_df</th><th>Vocabulary kept</th><th>% of vocabulary</th><th>% of tokens kept</th></tr>
<tr><td>1</td><td>19,001</td><td>100.0%</td><td>100.00%</td></tr>
<tr><td>2</td><td>10,871</td><td>57.2%</td><td>99.29%</td></tr>
<tr><td>3</td><td>8,782</td><td>46.2%</td><td>98.90%</td></tr>
<tr><td>5</td><td>6,885</td><td>36.2%</td><td>98.29%</td></tr>
<tr><td>10</td><td>5,083</td><td>26.8%</td><td>97.20%</td></tr>
<tr><td>50</td><td>2,326</td><td>12.2%</td><td>91.42%</td></tr>
</table>

Moving from `min_df=1` to `min_df=2` discards 42.8 percent of the vocabulary and loses 0.71
percent of the tokens. That is the hapax tail measured in phase 3 F4, and it costs almost
nothing to remove.

**`min_df=5` is the chosen default.** It keeps 98.29 percent of the tokens while cutting the
feature space by 64 percent, from 19,001 to 6,885. Beyond that the curve steepens: `min_df=50`
would cost 8.6 percent of tokens for a further reduction that KMeans does not need. The choice
is made from this table, not copied from the reference.

## V4. The four representations

<table>
<tr><th>Representation</th><th>Shape (train)</th><th>Sparsity</th><th>Median nonzero per document</th><th>Memory</th></tr>
<tr><td>count_unigram</td><td>14,332 x 6,885</td><td>99.099%</td><td>48</td><td>10.7 MB</td></tr>
<tr><td>tfidf_unigram</td><td>14,332 x 6,885</td><td>99.099%</td><td>48</td><td>10.7 MB</td></tr>
<tr><td>tfidf_bigram</td><td>14,332 x 46,531</td><td>99.780%</td><td>79</td><td>17.6 MB</td></tr>
<tr><td>count_unigram_maxdf07</td><td>14,332 x 6,885</td><td>99.099%</td><td>48</td><td>10.7 MB</td></tr>
</table>

Adding bigrams multiplies the feature space by 6.8, from 6,885 to 46,531, while the median
document contributes only 79 nonzero entries instead of 48. Most bigrams are near unique, which
is why sparsity rises to 99.78 percent. Whether that extra dimensionality buys anything is a
phase 6 question, not an assumption.

The fourth row exists only to document V1 and is not carried into phase 6, since it is
identical to the first.

## What phase 6 inherits

* Four matrices saved to `data/processed/features`, train and test, with vocabularies
* A fixed train and test split saved alongside them, so every experiment uses the same rows
* `min_df=5`, `max_df=1.0`, chosen from V3 and V1 respectively
* A specific thing to check rather than a hope: whether raw count clusters are length bands
