# Phase 9 findings: visualisation

Produced by `scripts/make_cluster_figures.py`. Figures 14 to 18, plus
`reports/sentiment_by_cluster.csv`.

This phase delivers the visual outputs the brief asks for: word clouds per cluster at K = 2 and
K = 8. It also pairs each of them with a quantitative equivalent, and reports one clear
negative result.

## X1. Word clouds, delivered, and their limits shown rather than argued

Figures 14 and 15 are the deliverable. Each cloud is built from the **TF IDF centroid weights**
of its cluster, not by concatenating the cluster's raw text. That choice matters: concatenation
lets a handful of very long documents supply most of the tokens, which is the length confound
of phase 5 reappearing in the visualisation layer. The reference solution concatenates.

The clouds are genuinely readable. Cluster 5 is dominated by mohela, student, loan,
forbearance. Cluster 7 by gift, card, declined, amex. Cluster 0 by debt, collection, validation,
collector. Anyone can look at figure 15 and name the eight themes.

Figure 16 shows the same eight clusters as ranked horizontal bars, where each bar is how far a
term's centroid weight exceeds its average across all eight clusters. Putting the two side by
side makes the limitation concrete rather than theoretical. A word cloud cannot show:

* **The gap between first and second place.** Font size is scaled for legibility, not
  proportion, so a term twice as important does not appear twice as large.
* **Whether a term is characteristic or merely common.** "account" and "credit" are large in
  several clouds because they are frequent everywhere. The ranked chart subtracts the average
  centroid, so shared terms disappear and only what distinguishes the cluster remains.
* **Any number at all.** You cannot read a magnitude off a cloud.

Both formats are kept. The clouds communicate a theme at a glance to someone who will not read
a chart; the bars carry the evidence.

## X2. The projection, coloured two ways

Figure 17 shows the same 5,000 documents twice: coloured by KMeans cluster on the left, by true
product label on the right. TSNE on the 100 component SVD space, perplexity 30.

**The left panel is tidy by construction and proves nothing.** KMeans drew those boundaries, so
of course the colours form contiguous regions. Showing only that panel, which is the common
practice, would be circular.

The right panel is the honest picture. There is real structure: student loans occupy a distinct
region at the bottom, prepaid cards a separate island at the top, and both correspond to the
near pure clusters 5 and 7 measured in phase 7. But the centre of the plot is a continuous
smear of credit card, checking, money transfer and debt collection complaints with no clean
boundaries at all. That smear is the visual form of the 47.5 percent purity ceiling.

One caveat stated on the figure itself: TSNE preserves local neighbourhoods and distorts global
distance. The gap between two distant blobs carries no information and should not be read as
one theme being further from another.

## X3. The sentiment layer does not work, and the way it fails is informative

Phase 1 proposed splitting the brief's "happy, neutral, sad, angry" framing into a theme axis
from clustering and an intensity axis from VADER. The theme axis worked. **The intensity axis
does not, and the honest thing is to report that rather than quietly drop it.**

Every one of the 14,332 documents is a complaint filed with a financial regulator. VADER scores
the corpus at a mean compound of **(0.042)**, with **45.1 percent of complaints scoring
positive**. Negative numbers are shown in parentheses.

By theme:

<table>
<tr><th>Cluster</th><th>Theme</th><th>Mean compound</th><th>% positive</th></tr>
<tr><td>7</td><td>Card declines and gift cards</td><td>+0.267</td><td>66.3%</td></tr>
<tr><td>3</td><td>Credit report accuracy</td><td>+0.221</td><td>65.2%</td></tr>
<tr><td>4</td><td>Statutory dispute template</td><td>+0.141</td><td>56.9%</td></tr>
<tr><td>5</td><td>Student loan servicing</td><td>+0.140</td><td>54.3%</td></tr>
<tr><td>1</td><td>Loan and mortgage servicing</td><td>(0.008)</td><td>47.5%</td></tr>
<tr><td>2</td><td>Vehicle finance and dealers</td><td>(0.049)</td><td>42.7%</td></tr>
<tr><td>6</td><td>Banking and transfers</td><td>(0.241)</td><td>33.4%</td></tr>
<tr><td>0</td><td>Debt collection practices</td><td>(0.380)</td><td>23.4%</td></tr>
</table>

**What VADER is actually measuring is register, not grievance.** Three pieces of evidence:

**One.** The most positively scored complaint in the corpus, at the maximum possible +1.000,
begins: "The creditor as fiduciary reached good faith and fair dealings with agreements crafted
by a corporation's unlawful practice of law with the intent of unlawfully converting..." That
is a hostile legal accusation. VADER sees "good faith" and "fair dealings" and scores it as
maximally positive.

**Two.** Templated filings score **+0.058** on average while unique filings score **(0.044)**.
The credit repair boilerplate is polite and formal, full of respectfully, request, fair, assist,
resolve, and VADER reads politeness as pleasure.

**Three.** The ordering of the table above is almost exactly an ordering by formality. The
templated and procedural themes sit at the top; the themes where people describe being chased
by collectors or losing access to their money sit at the bottom. Cluster 0 at (0.380) is
genuinely the angriest register, and cluster 3 at +0.221 is genuinely the most polite one. VADER
is measuring that axis correctly. It is simply not the axis anyone wanted.

**Why it fails here.** VADER is a rule based lexicon tuned on social media, where intensity is
carried by punctuation, capitalisation, emoji and intensifiers. Formal complaint prose carries
intensity through narrative instead: a calm, chronological account of losing four thousand
dollars is devastating and contains no negative words at all. VADER cannot read that, and no
amount of tuning the threshold will make it.

**What would work instead**, recorded as future work rather than claimed: a model fine tuned on
complaint or review text, or a simpler proxy the corpus already contains. The bureau records
`company_response` and `timely`, which are outcome signals rather than sentiment guesses, and
would likely serve the underlying business question, which complaint types go badly for
consumers, far better than any lexicon score.

## X4. A note on testing

The other phases added unit tests alongside their modules. This one adds none, deliberately.
`scripts/make_cluster_figures.py` is presentation code: it loads models already tested in
phases 6 and 8, and its output is images, verified by looking at them. Asserting on pixel
buffers would add maintenance cost without catching anything real. The one piece of logic worth
testing, the centroid weight extraction, is a thin wrapper over `cluster_centers_` which
`tests/test_cluster.py` already covers through `top_terms` and `distinctive_terms`.

## Figures produced

<table>
<tr><th>File</th><th>Shows</th></tr>
<tr><td>14_wordclouds_k2.png</td><td>Word clouds for the K = 2 split</td></tr>
<tr><td>15_wordclouds_k8.png</td><td>Word clouds for the eight themes</td></tr>
<tr><td>16_top_terms_k8.png</td><td>The same eight themes as ranked magnitudes</td></tr>
<tr><td>17_tsne.png</td><td>5,000 documents, coloured by cluster and by true label</td></tr>
<tr><td>18_sentiment.png</td><td>VADER distribution and mean by theme</td></tr>
</table>
