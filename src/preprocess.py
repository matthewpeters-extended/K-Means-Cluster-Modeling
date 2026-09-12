"""Phase 4. Turn raw complaint narratives into a modelling corpus.

Every step here answers a defect measured in phase 2 and recorded in docs/data_defects.md.
Nothing is carried over from the reference solution on faith: the ASCII filtering step it uses
is deliberately absent, because D7 measured zero non ASCII characters in this corpus.
"""

from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from .dedupe import REDACTION

URL = re.compile(r"https?://\S+|www\.\S+")
EMAIL = re.compile(r"\S+@\S+")
NON_ALPHA = re.compile(r"[^a-z\s]")
MIN_TOKEN_LEN = 3
MIN_DOC_TOKENS = 5

_lemmatiser = WordNetLemmatizer()


@lru_cache(maxsize=200_000)
def _lemma(token: str) -> str:
    return _lemmatiser.lemmatize(token)


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    """NLTK English stopwords, plus artefacts of this specific corpus.

    Deliberately small. Domain terms such as 'credit' and 'account' are NOT removed here:
    frequency based filtering belongs in the vectoriser, where max_df can make that decision
    from the data rather than from our assumptions. See PLAN.md phase 5.
    """
    artefacts = {"xxxx", "xxx", "xx"}
    return frozenset(set(stopwords.words("english")) | artefacts)


def clean_text(text: str) -> str:
    """Strip redaction, symbols and digits, drop short words and stopwords, lemmatise."""
    if not isinstance(text, str):
        return ""
    t = REDACTION.sub(" ", text)          # D3: 81% of docs carry redaction runs
    t = URL.sub(" ", t)
    t = EMAIL.sub(" ", t)
    t = t.lower()
    t = NON_ALPHA.sub(" ", t)             # removes digits and punctuation in one pass
    stops = _stopwords()
    tokens = [_lemma(w) for w in t.split()
              if len(w) >= MIN_TOKEN_LEN and w not in stops]
    tokens = [w for w in tokens if w not in stops]   # lemmatising can reveal a stopword
    return " ".join(tokens)


def build_corpus(df: pd.DataFrame, text_col: str = "complaint_what_happened") -> tuple[pd.DataFrame, dict]:
    """Clean, then deduplicate exactly, then collapse near duplicates. Returns (frame, stats)."""
    from .dedupe import collapse_near_duplicates

    stats: dict[str, int] = {"input": len(df)}

    out = df.copy()
    out["clean_text"] = out[text_col].map(clean_text)
    out["n_tokens"] = out["clean_text"].str.split().str.len().fillna(0).astype(int)

    out = out[out["n_tokens"] >= MIN_DOC_TOKENS]
    stats["after_min_length"] = len(out)

    out = out.drop_duplicates(subset="clean_text")
    stats["after_exact_dedupe"] = len(out)

    # Detect near duplicates on the ORIGINAL narrative rather than the cleaned one. Stopword
    # removal cuts a document roughly in half, which shortens the shingle set and makes two
    # paraphrases of the same template look less similar than they really are. Comparing the
    # raw text also keeps these numbers directly comparable with the phase 2 audit.
    out, groups = collapse_near_duplicates(out.reset_index(drop=True), text_col)
    stats["after_near_dedupe"] = len(out)
    stats["template_groups"] = len(groups)
    stats["largest_template"] = int(out["template_size"].max()) if len(out) else 0
    return out.reset_index(drop=True), stats
