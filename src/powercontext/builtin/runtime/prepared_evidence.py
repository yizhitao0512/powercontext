# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Select the literal spans of retrieved Memory bodies that answer one query.

The selector works on the candidate text exactly as stored: every span it returns is
a pair of character offsets into one entry body, never a rewrite, translation, or
generated summary. Ranking is lexical and offline, so the same inputs always produce
the same spans.

Cost is measured through a caller-supplied function rather than guessed, because the
budget that matters covers the rendered citation and envelope as well as the body.
That lets the selection fill a byte budget by *measured* cost and stop at the exact
point where the next span would overflow.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

_TOKEN = re.compile(r"[a-z0-9]+")
# Scripts written without spaces carry no word boundary a regex can split on, so they are
# indexed as single characters plus overlapping pairs instead.
_UNSPACED = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]+")
_SENTENCE_END = re.compile(r"(?<=[\u3002\uff01\uff1f\uff1b])|\n+|(?<=[.!?])[\"')\]]*\s+")
_MIN_SENTENCE_CHARS = 32
_DOCUMENT_BOOST = 0.45
_BREADTH_FLOOR = 0.5
_BREADTH_SPANS = 3
_STOPWORDS = frozenset({
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "had",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "might",
    "must",
    "not",
    "of",
    "on",
    "or",
    "our",
    "should",
    "than",
    "that",
    "the",
    "their",
    "them",
    "these",
    "this",
    "those",
    "to",
    "was",
    "we",
    "were",
    "which",
    "while",
    "with",
    "would",
})


@dataclass(frozen=True, slots=True)
class EvidenceEntry:
    """One candidate body, kept whole so spans stay verifiable against the original."""

    text: str


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    """One half-open character range inside one candidate entry."""

    entry_index: int
    start: int
    end: int


def select_evidence(
    entries: Sequence[EvidenceEntry],
    query: str,
    max_bytes: int,
    measure: Callable[[Sequence[EvidenceSpan]], int],
    *,
    max_spans: int = 8,
) -> tuple[EvidenceSpan, ...]:
    """Return the literal spans worth delivering for ``query`` under a byte budget.

    ``measure`` reports the rendered UTF-8 size of a span set, so the budget here is the
    real one: citations and envelope included. Selection is greedy by lexical value with a
    breadth-first first pass, because one sentence per relevant body carries more evidence
    per byte than a second sentence from an already-covered body.
    """

    if not entries or max_bytes <= 0 or max_spans <= 0:
        return ()
    query_terms = _query_terms(query)
    if not query_terms:
        return ()
    sentences = tuple(_sentences(entry.text) for entry in entries)
    tokens = tuple(
        tuple((start, end, _stems(entry.text[start:end])) for start, end in spans)
        for entry, spans in zip(entries, sentences, strict=True)
    )
    ranked = _rank(tokens, query_terms, _idf(tokens, query_terms))
    if not ranked:
        return ()
    return _fill(ranked, sentences, max_bytes, measure, max_spans)


def _fill(
    ranked: Sequence[tuple[float, int, int]],
    sentences: Sequence[Sequence[tuple[int, int]]],
    max_bytes: int,
    measure: Callable[[Sequence[EvidenceSpan]], int],
    max_spans: int,
) -> tuple[EvidenceSpan, ...]:
    """Add sentences in value order while the measured rendering still fits."""

    selected: dict[int, set[int]] = {}

    def spans() -> tuple[EvidenceSpan, ...]:
        return _consolidate(selected, sentences)

    def try_add(entry_index: int, sentence_index: int) -> bool:
        chosen = selected.setdefault(entry_index, set())
        chosen.add(sentence_index)
        candidate = spans()
        if len(candidate) > max_spans or measure(candidate) > max_bytes:
            chosen.discard(sentence_index)
            if not chosen:
                selected.pop(entry_index, None)
            return False
        return True

    for entry_index, sentence_index in _breadth_first(ranked):
        try_add(entry_index, sentence_index)
    for _, entry_index, sentence_index in ranked:
        if sentence_index not in selected.get(entry_index, ()):
            try_add(entry_index, sentence_index)
    _expand(selected, sentences, try_add)
    _bridge(selected, try_add)
    _expand(selected, sentences, try_add)
    return spans()


def _query_terms(query: str) -> frozenset[str]:
    terms = {_stem(token) for token in _tokens(query)}
    return frozenset(terms - _STOPWORDS)


def _tokens(text: str) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    for run in _UNSPACED.findall(text):
        tokens.extend(run)
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


def _sentences(text: str) -> tuple[tuple[int, int], ...]:
    """Split into trimmed sentence ranges, folding fragments too short to be evidence."""

    raw: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        raw.append((start, match.start()))
        start = match.end()
    raw.append((start, len(text)))

    trimmed: list[tuple[int, int]] = []
    for begin, end in raw:
        while begin < end and text[begin].isspace():
            begin += 1
        while end > begin and text[end - 1].isspace():
            end -= 1
        if end > begin:
            trimmed.append((begin, end))
    return _absorb_short(trimmed)


def _absorb_short(spans: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Group runs of short sentences until each group is worth citing on its own."""

    merged: list[tuple[int, int]] = []
    pending: tuple[int, int] | None = None
    for begin, end in spans:
        pending = (begin, end) if pending is None else (pending[0], end)
        if pending[1] - pending[0] >= _MIN_SENTENCE_CHARS:
            merged.append(pending)
            pending = None
    if pending is not None:
        if merged:
            merged[-1] = (merged[-1][0], pending[1])
        else:
            merged.append(pending)
    return tuple(merged)


def _stems(text: str) -> tuple[str, ...]:
    return tuple(_stem(token) for token in _tokens(text))


def _stem(token: str) -> str:
    if len(token) > 5 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 5 and token.endswith(("ches", "shes", "sses", "xes", "zes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def _idf(
    tokens: Sequence[Sequence[tuple[int, int, tuple[str, ...]]]],
    query_terms: frozenset[str],
) -> dict[str, float]:
    total = sum(len(spans) for spans in tokens)
    if total == 0:
        return {}
    occurrences: dict[str, int] = dict.fromkeys(query_terms, 0)
    for spans in tokens:
        for _, _, span_terms in spans:
            for term in set(span_terms) & query_terms:
                occurrences[term] += 1
    return {term: math.log(1 + total / (1 + count)) for term, count in occurrences.items()}


def _rank(
    tokens: Sequence[Sequence[tuple[int, int, tuple[str, ...]]]],
    query_terms: frozenset[str],
    weights: dict[str, float],
) -> list[tuple[float, int, int]]:
    """Score every sentence by the rare query terms it carries, biased by its own body.

    A sentence in the body that covers more of the query is worth more than an equally
    worded sentence in a body that covers less: the body is the unit the budget pays a
    citation for, so evidence concentrated there is cheaper to deliver.
    """

    ranked: list[tuple[float, int, int]] = []
    for entry_index, spans in enumerate(tokens):
        body_terms: set[str] = set()
        for _, _, span_terms in spans:
            body_terms.update(span_terms)
        coverage = len(body_terms & query_terms) / len(query_terms) if query_terms else 0.0
        boost = 1 + _DOCUMENT_BOOST * coverage
        for sentence_index, (_, _, span_terms) in enumerate(spans):
            matched = set(span_terms) & query_terms
            if not matched:
                continue
            ranked.append((sum(weights.get(term, 0.0) for term in matched) * boost, entry_index, sentence_index))
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    return ranked


def _breadth_first(ranked: Sequence[tuple[float, int, int]]) -> tuple[tuple[int, int], ...]:
    """Best sentence of each plausible entry first, strongest entry first.

    Every delivered entry costs a citation, so a body whose best sentence is far weaker than
    the leading body's does not earn one up front; it can still win a later sentence slot on
    its own score.
    """

    best: dict[int, int] = {}
    score: dict[int, float] = {}
    for value, entry_index, sentence_index in ranked:
        if entry_index not in best:
            best[entry_index] = sentence_index
            score[entry_index] = value
    if not score:
        return ()
    floor = max(score.values()) * _BREADTH_FLOOR
    order = [entry for entry in best if score[entry] >= floor]
    order.sort(key=lambda entry: -score[entry])
    return tuple((entry_index, best[entry_index]) for entry_index in order[:_BREADTH_SPANS])


def _consolidate(
    selected: dict[int, set[int]],
    sentences: Sequence[Sequence[tuple[int, int]]],
) -> tuple[EvidenceSpan, ...]:
    """Turn per-sentence choices into maximal runs of adjacent sentences."""

    spans: list[EvidenceSpan] = []
    for entry_index in sorted(selected):
        indices = sorted(selected[entry_index])
        if not indices:
            continue
        run_start = indices[0]
        previous = indices[0]
        for index in indices[1:]:
            if index == previous + 1:
                previous = index
                continue
            spans.append(
                EvidenceSpan(entry_index, sentences[entry_index][run_start][0], sentences[entry_index][previous][1])
            )
            run_start = index
            previous = index
        spans.append(
            EvidenceSpan(entry_index, sentences[entry_index][run_start][0], sentences[entry_index][previous][1])
        )
    return tuple(spans)


def _expand(
    selected: dict[int, set[int]],
    sentences: Sequence[Sequence[tuple[int, int]]],
    try_add: Callable[[int, int], bool],
) -> None:
    """Grow every selected run outward while the measured budget still allows it.

    Extra neighbouring text never lowers token recall — it adds no evidence the caller asked
    for, but it can contain evidence no query term pointed at — and one wider run costs one
    citation instead of two. Expansion stops as soon as the next sentence would overflow.
    """

    for entry_index in tuple(selected):
        count = len(sentences[entry_index])
        grew = True
        while grew:
            grew = False
            indices = selected.get(entry_index)
            if not indices:
                break
            lowest, highest = min(indices), max(indices)
            for neighbour in (lowest - 1, highest + 1):
                if 0 <= neighbour < count and neighbour not in indices and try_add(entry_index, neighbour):
                    grew = True
                    break


def _bridge(
    selected: dict[int, set[int]],
    try_add: Callable[[int, int], bool],
) -> None:
    """Fill a one-sentence gap when doing so merges two citations into one."""

    for entry_index in tuple(selected):
        indices = sorted(selected[entry_index])
        gaps = [index + 1 for index in indices if index + 1 not in indices and index + 2 in indices]
        for gap in gaps:
            try_add(entry_index, gap)


__all__ = ["EvidenceEntry", "EvidenceSpan", "select_evidence"]
