"""Loading, cleaning, de-duplication and splitting of the resume dataset.

The raw file has 13,389 OCR'd resumes labelled with one of 43 job titles.
Two things in it would quietly inflate every metric if left alone:

1. Duplicates. The same resume appears more than once (scraped from several
   sources). If one copy lands in train and another in test, the model is
   graded on text it has already memorised. We de-duplicate on a normalised
   form of the text *before* splitting.

2. Title leakage. Many resumes state their own job title ("Java Developer").
   A model can score well by spotting that phrase instead of reading the
   candidate's experience. `mask_title_words` removes the label's own words
   so we can measure how much the model leans on them.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

MIN_WORDS = 30
_non_alnum = re.compile(r"[^a-z0-9+#.]+")


def normalise(text: str) -> str:
    """Lower-case, collapse anything that is not a word character into spaces."""
    text = (text or "").lower().replace("﻿", " ")
    return " ".join(_non_alnum.sub(" ", text).split())


def fingerprint(text: str) -> str:
    return hashlib.sha1(normalise(text).encode()).hexdigest()


@dataclass
class CleanReport:
    raw_rows: int
    too_short: int
    duplicate_rows: int
    conflicting_label_groups: int
    kept_rows: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def load_clean(path: str) -> tuple[pd.DataFrame, CleanReport]:
    """Return de-duplicated resumes with a normalised `text` column."""
    raw = pd.read_csv(path)
    df = raw.rename(columns={"Category": "role", "Text": "raw_text"}).copy()
    df["raw_text"] = df["raw_text"].fillna("")
    df["text"] = df["raw_text"].map(normalise)
    df["n_words"] = df["text"].str.split().str.len()

    short = df["n_words"] < MIN_WORDS
    df = df[~short].copy()

    df["fp"] = df["text"].map(lambda t: hashlib.sha1(t.encode()).hexdigest())
    # A fingerprint carrying two different labels is ambiguous: drop all copies.
    labels_per_fp = df.groupby("fp")["role"].nunique()
    conflicting = labels_per_fp[labels_per_fp > 1].index
    n_conflict_rows = int(df["fp"].isin(conflicting).sum())
    df = df[~df["fp"].isin(conflicting)]
    before = len(df)
    df = df.drop_duplicates("fp").reset_index(drop=True)

    report = CleanReport(
        raw_rows=len(raw),
        too_short=int(short.sum()),
        duplicate_rows=before - len(df) + n_conflict_rows,
        conflicting_label_groups=len(conflicting),
        kept_rows=len(df),
    )
    return df[["role", "text", "n_words"]], report


def split(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70 / 15 / 15 train / validation / test split."""
    train, rest = train_test_split(df, test_size=0.30, stratify=df["role"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["role"], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def title_words(role: str) -> set[str]:
    return set(normalise(role).split()) - {"and"}


def mask_title_words(texts: pd.Series, roles: list[str]) -> pd.Series:
    """Remove every word that appears in ANY role name, so no label can be read off its own title."""
    banned = set().union(*(title_words(r) for r in roles))
    # also strip the obvious plural and the common short forms of those titles
    banned |= {w + "s" for w in banned} | {"dev", "devops", ".net", "sap", "hr", "qa"}
    return texts.map(lambda t: " ".join(w for w in t.split() if w not in banned))


def class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    return counts.sum() / (n_classes * np.maximum(counts, 1))
