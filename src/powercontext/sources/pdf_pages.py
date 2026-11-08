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

"""Full-document PDF parsing exposing physical pages as addressable evidence."""

from __future__ import annotations

import hashlib
import io
import math
import re
from collections.abc import Sequence
from urllib.parse import quote

from pydantic import BaseModel
from pypdf import PdfReader

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PDF_PAGE_SOURCE_NAME = "pdf-page"
DEFAULT_SEARCH_LIMIT = 10

_SEARCH_WORD = re.compile(r"[a-z]+|\d+")
# A financial question is mostly framing. Ranking a page on words every page carries rewards
# the pages that happen to be early in the document, so the framing is removed before scoring.
_SEARCH_FRAMING_WORDS = frozenset({
    "a", "an", "the", "and", "or", "for", "of", "to", "in", "on", "at", "by", "from", "with", "as", "is",
    "are", "was", "were", "be", "been", "being", "what", "which", "who", "how", "why", "when", "where", "do",
    "does", "did", "can", "could", "should", "would", "i", "we", "you", "it", "its", "our", "your", "this",
    "that", "these", "those", "please", "any", "has", "have", "had", "if", "then", "there", "their", "only",
    "use", "using", "used", "within", "give", "given", "response", "question", "relying", "detail", "details",
    "show", "shown", "state", "stated", "outline", "outlined", "explicitly", "not", "all", "also", "per",
    "answer", "answers", "respond", "assume", "assuming", "perspective", "analyst", "analysts", "define",
    "defined", "definition", "mention", "mentioned", "describe", "described", "according", "provide",
    "provided", "important", "ongoing", "report", "reported", "reporting", "quantity", "quantities",
    "exactly", "specifically", "generally", "overall", "currently", "recent", "latest", "key", "main",
    "major", "primarily", "primary", "directly", "fully", "roughly", "approximately",
})  # fmt: skip
_SEARCH_MIN_TERM_LENGTH = 3
# "geographies" never matches "geographic" through a tokenizer without stemming, so a term is
# also credited through the prefix both spellings share.
_SEARCH_PREFIX_LENGTH = 6
# The page opening carries the section heading, so a term found there identifies the topic.
_SEARCH_RUNNING_HEADER = "table of contents"
_SEARCH_HEADING_LINES = 6
# A section heading is how a filing names its own subject: "Item 3. Legal Proceedings" answers a
# question about legal matters even when the phrase in the question never appears in the body.
_SEARCH_HEADING_LINE = re.compile(
    r"item\s+\d+[a-z]?\.?\s+[A-Za-z][^.\n]{2,60}"
    r"|notes?\s+\d+\.?\s+[A-Za-z][^.\n]{2,60}"
    r"|consolidated\s+statements?\s+of\s+[a-z][a-z ]{2,40}"
    r"|consolidated\s+balance\s+sheets?",
    re.IGNORECASE,
)
_SEARCH_MAX_HEADINGS = 20
_SEARCH_HEADING_CHARS = 900
_SEARCH_HEADING_BOOST = 2.5
_SEARCH_TERM_QUERIES = 8
_SEARCH_CANDIDATE_LIMIT = 300
# Pages that run on from a strong page are strong candidates of their own.
_SEARCH_MIN_KEEP = 6
_SEARCH_NEIGHBOUR_DISTANCE = 3
# The page next to a matched page is as strong as the matched page; two pages away is nearly as
# strong, because a statement's table breaks across a page boundary.
_SEARCH_NEIGHBOUR_BONUS = {1: 2.2, 2: 2.2, 3: 1.5}
_SEARCH_NEIGHBOUR_WINDOW = 80


class PDFDocumentInput(BaseModel):
    """One complete original PDF together with the caller-assigned document id."""

    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    """One physical PDF page with the document digest and physical page number."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    """Captured physical page that replays through the pdf-page Definition."""

    page: PDFPageInput


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Build the revision-aware identity for one physical PDF page."""

    identity = f"pdf:{quote(document_id, safe='')}:{document_sha256}:{page_number}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"source identity must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return identity


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Parse every physical page of the original PDF in order, all or nothing."""

    document_id = value.document_id
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-empty string")
    content = value.content
    if not content:
        raise ValueError("PDF content must not be empty")

    document_sha256 = hashlib.sha256(content).hexdigest()
    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF requires a non-empty password")
        pages: list[PDFPageInput] = []
        for page_number, page in enumerate(reader.pages, start=1):
            pages.append(
                PDFPageInput(
                    document_id=document_id,
                    document_sha256=document_sha256,
                    page_number=page_number,
                    text=page.extract_text() or "",
                )
            )
        if not pages:
            raise ValueError("PDF must contain at least one page")
        return tuple(pages)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("PDF could not be parsed completely") from error


class PDFPageSourceAdapter:
    """Resolve and read PDF page Sources from parsed pages."""

    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        return PDFPageSource(
            name=pdf_page_source_id(value.document_id, value.document_sha256, value.page_number),
            materialization=SourceMaterialization.CAPTURED,
            page=value,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return source.page


class PDFPageTextEvidenceProjection:
    """Expose one physical PDF page as canonical textual evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        page = source.page
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_NAME,
            source_id=source.name,
            content=f"{page.document_id}\n{page.text}",
            metadata={
                "document_id": page.document_id,
                "document_sha256": page.document_sha256,
                "page_number": page.page_number,
            },
        )


def _search_terms(question: str) -> tuple[str, ...]:
    """Keep the content words of a question in the order they were asked."""

    terms: list[str] = []
    for word in _SEARCH_WORD.findall(question.lower()):
        if len(word) < _SEARCH_MIN_TERM_LENGTH or word in _SEARCH_FRAMING_WORDS or word in terms:
            continue
        terms.append(word)
    return tuple(terms)


def _page_terms(text: str) -> frozenset[str]:
    return frozenset(_SEARCH_WORD.findall(text.lower()))


def _page_heading(text: str) -> str:
    """Return the part of a page that names its subject.

    A report repeats its own header on every page, so taking the raw opening of the text
    credits every page with the company name and the fiscal year. What is left is the page
    opening plus every section heading on the page: a page that starts in the middle of a
    table still announces "Item 3. Legal Proceedings" or "Consolidated Statement of Income"
    somewhere, and that heading is what the question is actually about.
    """

    body = text.partition("\n")[2] or text
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    while lines and lines[0].lower().startswith(_SEARCH_RUNNING_HEADER):
        lines.pop(0)
    headings = [match.group(0) for match in _SEARCH_HEADING_LINE.finditer(body)][:_SEARCH_MAX_HEADINGS]
    return "\n".join([*lines[:_SEARCH_HEADING_LINES], *headings])[:_SEARCH_HEADING_CHARS]


def _term_hits(term: str, page: frozenset[str], prefixes: frozenset[str]) -> int:
    """Count the page words that answer ``term``, exactly or through a shared prefix."""

    prefix = term[:_SEARCH_PREFIX_LENGTH]
    if len(term) < _SEARCH_PREFIX_LENGTH:
        return 1 if term in page else 0
    return (1 if term in page else 0) + (1 if prefix in prefixes else 0)


def _rank_pages(
    candidates: Sequence[tuple[object, str]],
    terms: tuple[str, ...],
) -> list[object]:
    """Order candidate pages by how much of the question each one answers.

    Every term counts once per query, weighted by how rare it is inside the candidate set, so a
    page that carries the discriminating words outranks a page that only repeats the company
    name and the fiscal year. A hit in the page opening counts for more than the same hit buried
    in a table, because a financial statement names its subject in its heading — that heading is
    what an analyst searches for, and it is the only part of the page that is about one topic.
    """

    prepared: list[tuple[object, frozenset[str], frozenset[str], frozenset[str], frozenset[str]]] = []
    for hit, text in candidates:
        page = _page_terms(text)
        head = _page_terms(_page_heading(text))
        prepared.append((
            hit,
            page,
            frozenset(word[:_SEARCH_PREFIX_LENGTH] for word in page),
            head,
            frozenset(word[:_SEARCH_PREFIX_LENGTH] for word in head),
        ))
    total = len(prepared) or 1
    frequency = {
        term: sum(1 for _, page, prefixes, _, _ in prepared if _term_hits(term, page, prefixes)) for term in terms
    }
    scored: list[tuple[float, int, object]] = []
    for index, (hit, page, prefixes, head, head_prefixes) in enumerate(prepared):
        value = 0.0
        matched = 0
        for term in terms:
            hits = _term_hits(term, page, prefixes)
            if not hits:
                continue
            matched += 1
            idf = math.log(1 + total / (1 + frequency[term]))
            weight = 1 + math.log(hits)
            if _term_hits(term, head, head_prefixes):
                weight *= _SEARCH_HEADING_BOOST
            value += idf * idf * weight
        if not matched:
            continue
        scored.append((value, index, hit))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored


async def _page_number(memory, hit) -> int | None:
    """Read the physical page ordinal back out of the page Source identity."""

    from powercontext.builtin.artifacts.memory.models import MemoryCitation
    from powercontext.builtin.runtime import GetMemoryEntryRequest

    citation = MemoryCitation(
        memory_ref=hit.memory_ref,
        entry_id=hit.entry_id,
        entry_version_id=hit.entry_version_id,
    )
    try:
        entry = (await memory.get(GetMemoryEntryRequest(citation=citation))).entry
    except Exception:
        return None
    for reference in entry.sources:
        _, _, number = reference.source_id.rpartition(":")
        if number.isdigit():
            return int(number)
    return None


async def _promote_neighbours(memory, scored: Sequence[tuple[float, int, object]], limit: int) -> list[object]:
    """Favour the pages that sit next to a page whose content already matched.

    A financial statement, a table and a legal note all run past the bottom of a page, so the
    page that answers a question is regularly the one next to the page whose heading matched.
    A wrong page only costs a citation the caller can ignore, while the right page missing
    from the window is unrecoverable, so the neighbours of a matched page are promoted rather
    than dropped. The strongest pages keep their content ranks; the spare slots go to their
    neighbours, so opening a window on a strong page never pushes another strong page out.
    """

    if len(scored) <= 1:
        return [hit for _, _, hit in scored]
    keep = max(_SEARCH_MIN_KEEP, (limit * 13) // 20)
    head = scored[:keep]
    anchors: list[int] = []
    for _, _, hit in head:
        number = await _page_number(memory, hit)
        if number is not None:
            anchors.append(number)
    if not anchors:
        return [hit for _, _, hit in scored]
    contested: list[tuple[float, int, object]] = []
    for offset, (score, index, hit) in enumerate(scored[keep:], start=keep):
        # Reading the page ordinal is a round trip per candidate, so only the pages close enough
        # to the window to be promoted are looked up; the rest keep their content ranks.
        if offset < _SEARCH_NEIGHBOUR_WINDOW:
            number = await _page_number(memory, hit)
            if number is not None:
                distance = min(abs(number - anchor) for anchor in anchors)
                if 0 < distance <= _SEARCH_NEIGHBOUR_DISTANCE:
                    score *= _SEARCH_NEIGHBOUR_BONUS[distance]
        contested.append((score, index, hit))
    contested.sort(key=lambda item: (-item[0], item[1]))
    room = max(0, limit - keep)
    return [
        *(hit for _, _, hit in head),
        *(hit for _, _, hit in contested[:room]),
        *(hit for _, _, hit in contested[room:]),
    ]


async def search_pdf_pages(memory, question: str, limit: int = DEFAULT_SEARCH_LIMIT) -> tuple[MemoryCitation, ...]:
    """Return up to ``limit`` distinct physical page citations for ``question``.

    A financial question rarely repeats the wording of the page that answers it, so one MATCH
    over the whole sentence retrieves almost nothing: the tokenizer does not stem, and a page
    that answers "where does the company operate" says "geographic" rather than "geographies".
    The page pool is therefore collected from the question and from each of its content words,
    then re-ranked as whole pages, because the page — not the sentence — is the unit a citation
    has to address.
    """

    from powercontext.builtin.artifacts.memory.models import MemoryCitation
    from powercontext.builtin.runtime import SearchMemoryRequest

    if limit <= 0:
        return ()
    terms = _search_terms(question)
    pool: dict[str, object] = {}
    for query in (question, *terms[:_SEARCH_TERM_QUERIES]):
        page = await memory.search(SearchMemoryRequest(query=query, mode="fts", limit=_SEARCH_CANDIDATE_LIMIT))
        for hit in page.hits:
            pool.setdefault(hit.entry_id, hit)
    if not pool:
        return ()
    candidates = [(hit, hit.text) for hit in pool.values()]
    scored = (
        _rank_pages(candidates, terms) if terms else [(0.0, index, hit) for index, (hit, _) in enumerate(candidates)]
    )
    ranked = await _promote_neighbours(memory, scored, limit)
    return tuple(
        MemoryCitation(
            memory_ref=hit.memory_ref,
            entry_id=hit.entry_id,
            entry_version_id=hit.entry_version_id,
        )
        for hit in ranked[:limit]
    )


PDF_PAGE_SOURCE_ADAPTER = PDFPageSourceAdapter()
PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(
    PDF_PAGE_SOURCE_ADAPTER,
    projections=(PDFPageTextEvidenceProjection(),),
)

__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDF_PAGE_SOURCE_NAME",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "PDFPageSourceAdapter",
    "PDFPageTextEvidenceProjection",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]
