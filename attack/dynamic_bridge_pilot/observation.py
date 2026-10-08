"""Parse the clean first search observation (the pilot's d1) into documents.

The canonical cue is appended after the WHOLE first observation (all top-k
results of the first search call; attack/runtime/benchmark_search.py), so the
pilot's d1 is that whole observation. Facts may only be extracted from the
SUMMARY text of each result, never from the metadata lines (URL, IDs, scores),
and every span keeps its absolute character offset into the observation, which
is the tie-break order tau.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADER_RE = re.compile(r"^--- SOURCE (?P<rank>\d+): (?P<title>.*) ---$", re.MULTILINE)
_DOCID_RE = re.compile(r"^DOCUMENT_ID: (?P<doc_id>.+)$", re.MULTILINE)
_SUMMARY_MARK = "\nSUMMARY:\n"
_RULE = "\n" + "-" * 80


@dataclass(frozen=True)
class ObservedDocument:
    position: int          # order of appearance in the observation (0-based)
    rank: int
    title: str
    document_id: str
    text: str
    text_start: int        # absolute offset of `text` in the observation
    text_end: int


def parse_observation(observation: str) -> list[ObservedDocument]:
    headers = list(_HEADER_RE.finditer(observation))
    docs: list[ObservedDocument] = []
    for i, h in enumerate(headers):
        block_end = headers[i + 1].start() if i + 1 < len(headers) else len(observation)
        block = observation[h.start():block_end]
        doc_id_m = _DOCID_RE.search(block)
        s = block.find(_SUMMARY_MARK)
        if s < 0:
            continue
        text_start = h.start() + s + len(_SUMMARY_MARK)
        rule = observation.find(_RULE, text_start, block_end)
        text_end = rule if rule >= 0 else block_end
        text = observation[text_start:text_end].rstrip("\n")
        docs.append(ObservedDocument(
            position=len(docs), rank=int(h.group("rank")), title=h.group("title"),
            document_id=doc_id_m.group("doc_id").strip() if doc_id_m else "",
            text=text, text_start=text_start, text_end=text_start + len(text),
        ))
    return docs


def locate_span(observation: str, docs: list[ObservedDocument], span: str) -> tuple[int, ObservedDocument] | None:
    """First occurrence of `span` verbatim inside some document's SUMMARY text.
    Returns (absolute offset, document) or None. Metadata never matches."""
    if not span or not span.strip():
        return None
    for doc in docs:
        j = doc.text.find(span)
        if j >= 0:
            return doc.text_start + j, doc
    return None


def documents_for_prompt(docs: list[ObservedDocument]) -> str:
    """The attacker sees document titles and text only (no IDs/URLs/scores)."""
    return "\n\n".join(f"[Document {d.position + 1}] {d.title}\n{d.text}" for d in docs)
