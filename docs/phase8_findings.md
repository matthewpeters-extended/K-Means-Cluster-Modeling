# Phase 8 findings: KMeans against NMF and LDA

Produced by `scripts/compare_methods.py`. Tables in `reports/method_comparison.csv` and
`reports/topics_k8.csv`. Figure 13.

The reference solution argues for KMeans over LDA and NMF on four grounds: simplicity, speed,
hard assignment and interpretability. It never fits either alternative. Here both are fitted on
the same corpus, the same split and the same vocabulary gates, and scored with the same metrics
as phase 7, so the argument can be settled rather than asserted.

One representation decision is not arbitrary. NMF runs on TF IDF, the standard choice for
Frobenius loss factorisation. LDA runs on raw counts because it must: it is a generative model
of how integer word counts arise, and TF IDF values are neither integers nor counts.

## M1. The length collapse was KMeans, not raw counts

This is the most useful thing phase 8 establishes, and it corrects a natural misreading of
phases 5 to 7.

Share of document length variance explained by the partition, on train:

<table>
<tr><th>K</th><th>KMeans (counts)</th><th>LDA (counts)</th><th>KMeans (tfidf)</th><th>NMF (tfidf)</th></tr>
<tr><td>2</td><td>42.6%</td><td>0.2%</td><td>0.0%</td><td>0.0%</td></tr>
<tr><td>8</td><td>61.5%</td><td>0.2%</td><td>13.9%</td><td>11.8%</td></tr>
<tr><td>11</td><td>61.2%</td><td>1.6%</td><td>12.8%</td><td>11.3%</td></tr>
</table>

**LDA consumes the identical raw count matrix that made KMeans collapse into length bands, and
is completely immune to the problem.** At K = 8 the figures are 61.5 percent against 0.2
percent.

So the defect diagnosed in phase 6 was never "raw counts are bad". It was the interaction
between raw counts and Euclidean distance. KMeans measures straight line distance in a space
where vector magnitude is document length, so length dominates. LDA never measures distance at
all: it models each document as a mixture drawn from a Dirichlet, which normalises the document
away by construction.

LDA is in fact the **least** length coupled method tested, lower than either TF IDF method.
Stating the phase 6 conclusion as "use TF IDF instead of counts" would have been true in
effect and wrong in mechanism.

## M2. Held out results against product labels

<table>
<tr><th>K</th><th>Method</th><th>Purity</th><th>Permuted</th><th>ARI</th><th>NMI</th></tr>
<tr><td>11</td><td>NMF (tfidf)</td><td>0.460</td><td>0.123</td><td><b>0.265</b></td><td>0.369</td></tr>
<tr><td>11</td><td>KMeans (tfidf)</td><td><b>0.467</b></td><td>0.123</td><td>0.233</td><td><b>0.386</b></td></tr>
<tr><td>11</td><td>LDA (counts)</td><td>0.351</td><td>0.120</td><td>0.141</td><td>0.275</td></tr>
<tr><td>11</td><td>KMeans (counts)</td><td>0.198</td><td>0.114</td><td>0.014</td><td>0.141</td></tr>
<tr><td>8</td><td>KMeans (tfidf)</td><td>0.384</td><td>0.119</td><td>0.195</td><td>0.337</td></tr>
<tr><td>8</td><td>NMF (tfidf)</td><td>0.367</td><td>0.118</td><td>0.195</td><td>0.301</td></tr>
<tr><td>8</td><td>LDA (counts)</td><td>0.283</td><td>0.118</td><td>0.124</td><td>0.241</td></tr>
<tr><td>8</td><td>KMeans (counts)</td><td>0.188</td><td>0.112</td><td>0.014</td><td>0.123</td></tr>
<tr><td>2</td><td>NMF (tfidf)</td><td>0.156</td><td>0.107</td><td>0.042</td><td>0.107</td></tr>
<tr><td>2</td><td>KMeans (tfidf)</td><td>0.153</td><td>0.107</td><td>0.039</td><td>0.119</td></tr>
<tr><td>2</td><td>LDA (counts)</td><td>0.147</td><td>0.107</td><td>0.029</td><td>0.083</td></tr>
<tr><td>2</td><td>KMeans (counts)</td><td>0.114</td><td>0.105</td><td>0.000</td><td>0.007</td></tr>
</table>

**NMF at K = 11 produces the best adjusted Rand index in the entire project, 0.265**, edging
past every KMeans configuration from phase 7. KMeans on TF IDF holds a narrow lead on purity
and NMI at the same K. The two are, for practical purposes, tied.

LDA is clearly third on every metric at every K.

## M3. The verdict on the reference solution's argument

Taking its four claims in turn.

**Speed. Half right.** KMeans fits in 1.0 seconds at K = 11. LDA takes 19.0 seconds, so the
claim holds by a factor of 19 against LDA. Against NMF it does not: NMF fits in 1.3 seconds.
The reference groups NMF with LDA as the slow alternatives, and on this corpus NMF is
essentially as fast as KMeans.

**Accuracy. Right by accident.** KMeans does perform at the top, but tied with NMF rather than
ahead of it, and the reference offers no evidence because it never fitted the comparison. More
importantly, the gap between KMeans on TF IDF and KMeans on counts is a factor of roughly 14 on
ARI, while the gap between KMeans and NMF is nil. **The representation mattered far more than
the algorithm**, and the reference gets the algorithm choice right while getting the
representation choice wrong.

**Hard assignment. Genuinely vindicated, and the strongest of the four claims.** Measuring how
decisive the soft models actually are:

<table>
<tr><th>Method</th><th>K</th><th>Mean weight on top topic</th><th>Uniform would be</th><th>Documents above 0.5</th></tr>
<tr><td>NMF</td><td>8</td><td>0.514</td><td>0.125</td><td>46.2%</td></tr>
<tr><td>LDA</td><td>8</td><td>0.621</td><td>0.125</td><td>70.4%</td></tr>
<tr><td>NMF</td><td>11</td><td>0.473</td><td>0.091</td><td>37.1%</td></tr>
<tr><td>LDA</td><td>11</td><td>0.569</td><td>0.091</td><td>58.7%</td></tr>
</table>

At K = 11, NMF places more than half its weight on a single topic for only 37 percent of
documents. For the other 63 percent the argmax label is a weak claim that discards most of the
model's actual output. So the reference's point stands: if a business wants one label per
complaint, KMeans gives one honestly, while taking the argmax of a hedged distribution
manufactures false confidence.

**Interpretability. Contested.** NMF produces the most balanced partition of anything tested,
and LDA produces degenerate topics.

<table>
<tr><th>Method</th><th>Largest topic</th><th>Smallest topic</th><th>Size ratio</th><th>Normalised entropy</th></tr>
<tr><td>NMF (tfidf)</td><td>19.4%</td><td>8.1%</td><td><b>2.4</b></td><td><b>0.978</b></td></tr>
<tr><td>KMeans (tfidf)</td><td>23.6%</td><td>5.2%</td><td>4.6</td><td>0.910</td></tr>
<tr><td>LDA (counts)</td><td>25.2%</td><td>0.1%</td><td>240.8</td><td>0.841</td></tr>
<tr><td>KMeans (counts)</td><td>68.9%</td><td>0.0%</td><td>2,467.5</td><td>0.504</td></tr>
</table>

LDA at K = 8 puts 0.1 percent of documents in one topic, and at K = 11 one of its eleven topics
receives no documents at all as an argmax winner, leaving ten usable topics out of eleven
requested. NMF never does this. If balanced, usable topics are what interpretability means,
NMF is the better tool, not KMeans.

## M4. Similar scores, different structure

KMeans and NMF score almost identically, so it is tempting to assume they found the same thing.
They did not. Adjusted Rand index between the two partitions themselves:

* K = 8: 0.358
* K = 11: 0.365

**They agree with each other about as weakly as either agrees with the official taxonomy.**
Two methods reaching the same score by carving the corpus up differently is a reminder that
there is no single correct partition of this data, only several defensible ones.

The topic contents show it. NMF's eight topics include one built entirely around the mechanics
of contact: told, called, call, said, back, asked, phone, email, at 15.5 percent of the corpus.
That is a register cluster like the statutory template one from phase 7, describing how the
complaint is told rather than what it is about. KMeans does not produce it.

## M5. LDA's topics are the least coherent

LDA topic 4 at K = 8 mixes mortgage, card, insurance, gift, home, escrow, use, express,
american, tax, property. Mortgage escrow and American Express gift cards are not one subject.
Topic 2 holds 0.1 percent of documents and is about a single lender.

This is partly a fairness issue worth naming: LDA had 25 online iterations, and more would
likely help. It is also partly structural. LDA assumes documents are mixtures over topics with
a sparse Dirichlet prior, which suits long documents on multiple subjects. The median cleaned
complaint here is 66 tokens and is usually about exactly one thing. The generative story LDA
tells does not match how this corpus was written.

## Conclusion

* **NMF on TF IDF is the best method tested**, on the combination of the highest ARI in the
  project (0.265), the most balanced topics (size ratio 2.4) and speed comparable to KMeans.
* **KMeans on TF IDF is a very close second** and wins on purity, NMI and decisiveness of
  assignment.
* **LDA is third**, slower by a factor of 19, less accurate on every metric, and prone to
  degenerate topics on short documents.
* The reference solution's preference for KMeans is defensible, and it reached that conclusion
  without evidence. Its best argument, hard assignment, is the one it states least prominently.
* The representation choice outweighed the algorithm choice by roughly an order of magnitude.
  That is the lesson worth carrying to the next project.
