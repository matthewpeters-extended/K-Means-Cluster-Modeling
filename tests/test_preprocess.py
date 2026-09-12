"""Tests for the cleaning pipeline. Each one pins a decision made in docs/data_defects.md."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocess import MIN_DOC_TOKENS, build_corpus, clean_text


def test_strips_redaction_runs():
    # D3: runs of X mask names and account numbers and must not reach the vocabulary
    assert "xxxx" not in clean_text("I called XXXX on XX/XX/2024 about my account")
    assert "xx" not in clean_text("Payment of $XXXX was taken").split()


def test_strips_digits_and_punctuation():
    out = clean_text("They charged me $1,250.99 on 03/04/2024 -- twice!")
    assert not any(ch.isdigit() for ch in out)
    assert all(ch.isalpha() or ch == " " for ch in out)


def test_drops_short_words_and_stopwords():
    out = clean_text("I am so very upset at the bank").split()
    assert "am" not in out and "so" not in out   # under three characters
    assert "the" not in out and "at" not in out  # stopwords
    assert "upset" in out and "bank" in out


def test_lemmatises():
    out = clean_text("accounts were charged fees by agencies").split()
    assert "account" in out and "fee" in out and "agency" in out


def test_keeps_domain_vocabulary():
    # Frequency filtering is the vectoriser's job, not the cleaner's. See PLAN.md phase 5.
    out = clean_text("my credit account with the mortgage company").split()
    assert "credit" in out and "account" in out and "mortgage" in out


def test_strips_urls_and_emails():
    out = clean_text("see https://example.com or write to me@example.com please")
    assert "http" not in out and "example" not in out and "com" not in out


@pytest.mark.parametrize("bad", [None, "", "   ", 12345, float("nan")])
def test_survives_junk_input(bad):
    assert clean_text(bad) == "" or isinstance(clean_text(bad), str)


def _frame(texts):
    return pd.DataFrame({
        "complaint_what_happened": texts,
        "product": ["Credit card"] * len(texts),
    })


def test_removes_documents_that_are_too_short():
    df = _frame(["the a of and", "my credit card account was charged incorrectly twice this month"])
    out, stats = build_corpus(df)
    assert stats["after_min_length"] == 1
    assert (out["n_tokens"] >= MIN_DOC_TOKENS).all()


def test_exact_duplicates_are_removed():
    text = "my credit card account was charged incorrectly twice this month"
    out, stats = build_corpus(_frame([text, text, text]))
    assert stats["after_exact_dedupe"] == 1


def test_near_duplicates_are_collapsed_and_counted():
    # D2: templated filings differ only in wording, so exact matching misses them
    base = ("In accordance with the Fair Credit Reporting Act this creditor has violated my "
            "rights under 15 USC 1681 section 602 and I demand the account be removed from my "
            "consumer report immediately without any further delay whatsoever")
    variants = [base,
                base.replace("immediately", "at once"),
                base.replace("whatsoever", "at all"),
                base.replace("demand", "insist")]
    unrelated = ("my mortgage servicer lost the escrow payment and the property tax bill went "
                 "unpaid for several months causing penalties to accrue on the loan")
    out, stats = build_corpus(_frame(variants + [unrelated]))

    assert stats["after_exact_dedupe"] == 5, "the variants are not exact duplicates"
    assert stats["after_near_dedupe"] == 2, "the four variants should collapse to one"
    assert out["template_size"].max() == 4, "the collapsed group must record what it stood for"
    assert out["template_size"].min() == 1, "the unrelated document stands alone"


def test_no_rows_are_invented():
    df = _frame(["my credit card account was charged incorrectly twice this month"] * 2
                + ["the mortgage servicer lost my escrow payment and taxes went unpaid"])
    out, stats = build_corpus(df)
    assert len(out) <= stats["input"]
    assert out["template_size"].sum() <= stats["input"]
