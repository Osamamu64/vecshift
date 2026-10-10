"""Test queries for comparing the old and new vectors, in any mix of Arabic and Latin text."""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

ARABIC = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
LATIN = re.compile(r"[A-Za-z\u00C0-\u024F]")
# Sentence ends in Latin and Arabic text: . ! ? and the Arabic question mark and full stop.
_SENTENCE = re.compile(r"(?<=[.!?\u061F\u06D4])\s+|\n+")
_WORDS = re.compile(r"\S+")

SCRIPTS = ("arabic", "latin", "other")


def script(text: str) -> str:
    """``arabic``, ``latin``, or ``other``: the writing system most of the letters use.

    It's a script, not a language: English and French are both ``latin``. That's enough to
    show whether a model handles Arabic, without a language-detection model.
    """
    arabic = len(ARABIC.findall(text))
    latin = len(LATIN.findall(text))
    if not arabic and not latin:
        return "other"
    return "arabic" if arabic >= latin else "latin"


@dataclass(frozen=True, slots=True)
class EvalQuery:
    text: str
    relevant: frozenset[str]
    """Keys of the rows that answer this query."""
    language: str
    """Script of the query."""
    doc_language: str
    """Script of the row that answers it."""

    @property
    def slice(self) -> str:
        """Results are grouped by query and document script, e.g. ``arabic→latin``."""
        return f"{self.language}→{self.doc_language}"


def _norm(sentence: str) -> str:
    return " ".join(sentence.lower().split())


def proxy_queries(
    rows: Sequence[tuple[str, str]], n: int, seed: int
) -> tuple[list[EvalQuery], list[str]]:
    """Make queries from the rows themselves, with no LLM.

    One sentence of a row becomes a query whose answer is that row. The sentence is still
    part of the stored vector's text, so these are easy known-item searches: good for
    comparing the old and new vectors, generous as absolute numbers. Returns the queries
    and notes for the report.
    """
    rng = random.Random(seed)
    split = {key: [s.strip() for s in _SENTENCE.split(text) if s.strip()] for key, text in rows}
    # Sentences repeated across rows (boilerplate) have more than one right answer.
    counts = Counter(_norm(s) for sentences in split.values() for s in set(sentences))
    candidates: dict[str, tuple[str, str]] = {}
    for key, text in rows:
        usable = [
            s for s in split[key] if 5 <= len(_WORDS.findall(s)) <= 40 and counts[_norm(s)] == 1
        ]
        if usable:
            candidates[key] = (rng.choice(usable), text)
            continue
        words = _WORDS.findall(text)
        if len(words) >= 12:
            candidates[key] = (" ".join(words[: min(12, len(words) // 2)]), text)
    chosen = sorted(candidates)
    rng.shuffle(chosen)
    chosen = chosen[:n]
    queries = [
        EvalQuery(
            candidates[key][0],
            frozenset({key}),
            script(candidates[key][0]),
            script(candidates[key][1]),
        )
        for key in chosen
    ]
    notes = [
        "Queries are sentences taken from the rows themselves (no LLM). The sentence is "
        "still in the row's text, so absolute scores run high; the comparison between old "
        "and new is what counts. Use --generate-queries or --queries for realistic numbers."
    ]
    if len(queries) < n:
        notes.append(f"Only {len(queries)} rows had text long enough to make queries from.")
    return queries, notes
