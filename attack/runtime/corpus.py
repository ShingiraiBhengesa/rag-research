"""Dataset corpus and evidence builders for benchmark-local retrieval.

Corpus rows are safe for the model/search tool. Evidence rows are evaluation-only
and must never be mixed into corpus rows.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

from attack.runtime.evidence import EvidenceDependency
from attack.runtime.hashing import hash_text, stable_hash


MUSIQUE_SCOPE_NOTE = (
    "This attack supports arbitrary MuSiQue decomposition lengths and evaluates the "
    "unfiltered sampled hop distribution. Hop-specific outcomes are reported separately."
)


@dataclass(frozen=True)
class CorpusRow:
    dataset: str
    example_id: str
    corpus_id: str
    document_id: str
    title: str
    text: str
    paragraph_id: str = ""

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceRow:
    dataset: str
    example_id: str
    question: str
    answer: str
    dependencies: tuple[EvidenceDependency, ...]
    answer_aliases: tuple[str, ...] = ()
    hop_count: int = 0
    limitation: str = ""

    def to_json(self) -> dict:
        row = asdict(self)
        row["dependencies"] = [dep.to_json() for dep in self.dependencies]
        return row


@dataclass(frozen=True)
class MultiHopRAGAudit:
    example_id: str
    evidence_entry_count: int
    has_stable_document_ids: bool
    evidence_appears_gold_only: bool
    can_define_oracle_complete: bool
    answer_in_annotated_evidence: bool

    def to_json(self) -> dict:
        return asdict(self)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def stable_document_id(dataset: str, example_id: str, title: str, index: int) -> str:
    title_key = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "untitled"
    digest = stable_hash({"dataset": dataset, "example_id": example_id, "title": title, "index": index}, length=12)
    return f"{dataset}:{example_id}:{index}:{title_key}:{digest}"


def stable_dependency_id(dataset: str, example_id: str, hop_index: int, supporting_paragraph_id: str) -> str:
    digest = stable_hash(
        {
            "dataset": dataset,
            "example_id": example_id,
            "hop_index": hop_index,
            "supporting_paragraph_id": supporting_paragraph_id,
        },
        length=12,
    )
    return f"{dataset}:{example_id}:hop_{hop_index}:{supporting_paragraph_id}:{digest}"


def build_hotpot_rows(ex: Mapping[str, object]) -> tuple[list[CorpusRow], EvidenceRow]:
    dataset = "hotpot"
    example_id = str(ex["id"])
    titles = [str(title) for title in ex["context"]["title"]]
    sent_lists = ex["context"]["sentences"]
    docs = []
    for idx, title in enumerate(titles):
        doc_id = stable_document_id(dataset, example_id, title, idx)
        docs.append(
            CorpusRow(
                dataset=dataset,
                example_id=example_id,
                corpus_id=doc_id,
                document_id=doc_id,
                title=_normalize(title),
                text=_normalize(" ".join(sent_lists[idx])),
                paragraph_id=doc_id,
            )
        )
    doc_by_title = {doc.title: doc for doc in docs}
    dependencies = []
    final_answer = _normalize(str(ex["answer"]))
    for dep_idx, title in enumerate(dict.fromkeys(ex["supporting_facts"]["title"])):
        title_norm = _normalize(str(title))
        doc = doc_by_title.get(title_norm)
        if doc is None:
            continue
        sent_ids = tuple(
            int(sentence_id)
            for sf_title, sentence_id in zip(ex["supporting_facts"]["title"], ex["supporting_facts"]["sent_id"])
            if _normalize(str(sf_title)) == title_norm
        )
        dependencies.append(
            EvidenceDependency(
                dependency_id=f"hotpot:{example_id}:support:{dep_idx}",
                document_id=doc.document_id,
                title=doc.title,
                hop_index=dep_idx,
                sentence_ids=sent_ids,
                supporting_paragraph_id=doc.paragraph_id,
                answer_bearing_spans=(final_answer,) if final_answer and final_answer.lower() in doc.text.lower() else (),
                decomposition_step=dep_idx,
            )
        )
    return docs, EvidenceRow(
        dataset=dataset,
        example_id=example_id,
        question=_normalize(str(ex["question"])),
        answer=final_answer,
        answer_aliases=(),
        dependencies=tuple(dependencies),
        hop_count=len(dependencies),
    )


def build_2wiki_rows(ex: Mapping[str, object]) -> tuple[list[CorpusRow], EvidenceRow]:
    dataset = "2wiki"
    example_id = str(ex["_id"])
    docs = []
    for idx, (title, sentences) in enumerate(ex["context"]):
        title_norm = _normalize(str(title))
        doc_id = stable_document_id(dataset, example_id, title_norm, idx)
        docs.append(
            CorpusRow(
                dataset=dataset,
                example_id=example_id,
                corpus_id=doc_id,
                document_id=doc_id,
                title=title_norm,
                text=_normalize(" ".join(sentences)),
                paragraph_id=doc_id,
            )
        )
    doc_by_title = {doc.title: doc for doc in docs}
    dependencies = []
    final_answer = _normalize(str(ex["answer"]))
    for dep_idx, title in enumerate(dict.fromkeys(title for title, _sent_id in ex["supporting_facts"])):
        title_norm = _normalize(str(title))
        doc = doc_by_title.get(title_norm)
        if doc is None:
            continue
        sent_ids = tuple(
            int(sentence_id)
            for sf_title, sentence_id in ex["supporting_facts"]
            if _normalize(str(sf_title)) == title_norm
        )
        dependencies.append(
            EvidenceDependency(
                dependency_id=f"2wiki:{example_id}:support:{dep_idx}",
                document_id=doc.document_id,
                title=doc.title,
                hop_index=dep_idx,
                sentence_ids=sent_ids,
                supporting_paragraph_id=doc.paragraph_id,
                answer_bearing_spans=(final_answer,) if final_answer and final_answer.lower() in doc.text.lower() else (),
                decomposition_step=dep_idx,
            )
        )
    return docs, EvidenceRow(
        dataset=dataset,
        example_id=example_id,
        question=_normalize(str(ex["question"])),
        answer=final_answer,
        answer_aliases=(),
        dependencies=tuple(dependencies),
        hop_count=len(dependencies),
    )


def build_musique_rows(ex: Mapping[str, object]) -> tuple[list[CorpusRow], EvidenceRow]:
    """Build MuSiQue rows from all official decomposition steps.

    Official `question_decomposition` order defines hop order. Each step's
    `paragraph_support_idx` identifies the supporting paragraph used for
    evaluation-only coverage labels; BM25 never receives that label.
    """
    dataset = "musique"
    example_id = str(ex["id"])
    decomp = ex["question_decomposition"]
    docs = []
    for idx, paragraph in enumerate(ex["paragraphs"]):
        title = _normalize(str(paragraph["title"]))
        paragraph_id = _paragraph_id(paragraph, idx)
        docs.append(
            CorpusRow(
                dataset=dataset,
                example_id=example_id,
                corpus_id=stable_document_id(dataset, example_id, title, idx),
                document_id=stable_document_id(dataset, example_id, title, idx),
                title=title,
                text=_normalize(str(paragraph["paragraph_text"])),
                paragraph_id=paragraph_id,
            )
        )
    dependencies = []
    final_answer = _normalize(str(ex["answer"]))
    answer_aliases = tuple(_normalize(str(alias)) for alias in ex.get("answer_aliases", []) if str(alias).strip())
    for dep_idx, step in enumerate(decomp):
        paragraph_idx = step.get("paragraph_support_idx")
        if paragraph_idx is None or int(paragraph_idx) >= len(docs):
            paragraph_id = "" if paragraph_idx is None else str(paragraph_idx)
            dependencies.append(
                EvidenceDependency(
                    dependency_id=stable_dependency_id(dataset, example_id, dep_idx, paragraph_id or "missing"),
                    document_id=f"missing-paragraph-{paragraph_idx}",
                    hop_index=dep_idx,
                    supporting_paragraph_id=paragraph_id,
                    decomposition_question=_normalize(str(step.get("question") or "")),
                    intermediate_answer=_normalize(str(step.get("answer") or "")),
                    decomposition_step=int(step.get("id") if step.get("id") is not None else dep_idx),
                    ambiguous=True,
                )
            )
            continue
        doc = docs[int(paragraph_idx)]
        paragraph = ex["paragraphs"][int(paragraph_idx)]
        paragraph_id = _paragraph_id(paragraph, int(paragraph_idx))
        intermediate_answer = _normalize(str(step.get("answer") or ""))
        answer_spans = (final_answer,) if final_answer and final_answer.lower() in doc.text.lower() else ()
        dependencies.append(
            EvidenceDependency(
                dependency_id=stable_dependency_id(dataset, example_id, dep_idx, paragraph_id),
                document_id=doc.document_id,
                title=doc.title,
                hop_index=dep_idx,
                supporting_paragraph_id=paragraph_id,
                decomposition_question=_normalize(str(step.get("question") or "")),
                intermediate_answer=intermediate_answer,
                bridge_entities=(intermediate_answer,) if intermediate_answer else (),
                answer_bearing_spans=answer_spans,
                decomposition_step=int(step.get("id") if step.get("id") is not None else dep_idx),
            )
        )
    return docs, EvidenceRow(
        dataset=dataset,
        example_id=example_id,
        question=_normalize(str(ex["question"])),
        answer=final_answer,
        answer_aliases=answer_aliases,
        dependencies=tuple(dependencies),
        hop_count=len(decomp),
        limitation=MUSIQUE_SCOPE_NOTE,
    )


def build_multihoprag_rows(ex: Mapping[str, object], *, row_index: int) -> tuple[list[CorpusRow], EvidenceRow, MultiHopRAGAudit]:
    dataset = "multihoprag"
    example_id = str(ex.get("id") or f"mhrag_{row_index}")
    evidence_list = list(ex.get("evidence_list") or [])
    docs = []
    dependencies = []
    answer = _normalize(str(ex.get("answer") or ""))

    for idx, evidence in enumerate(evidence_list):
        title = _normalize(str(evidence.get("title") or f"Evidence {idx}"))
        text = _normalize(str(evidence.get("fact") or evidence.get("text") or ""))
        doc_id = stable_document_id(dataset, example_id, title, idx)
        docs.append(
            CorpusRow(
                dataset=dataset,
                example_id=example_id,
                corpus_id=doc_id,
                document_id=doc_id,
                title=title,
                text=text,
                paragraph_id=doc_id,
            )
        )
        dependencies.append(
            EvidenceDependency(
                dependency_id=f"multihoprag:{example_id}:evidence:{idx}",
                document_id=doc_id,
                title=title,
                hop_index=idx,
                answer_bearing_spans=(answer,) if answer and answer.lower() in text.lower() else (),
                decomposition_step=idx,
                ambiguous=False,
            )
        )

    audit = MultiHopRAGAudit(
        example_id=example_id,
        evidence_entry_count=len(evidence_list),
        has_stable_document_ids=all(bool(doc.document_id) for doc in docs),
        evidence_appears_gold_only=True,
        can_define_oracle_complete=bool(evidence_list) and all(bool(doc.text) for doc in docs),
        answer_in_annotated_evidence=bool(answer) and any(answer.lower() in doc.text.lower() for doc in docs),
    )
    return docs, EvidenceRow(
        dataset=dataset,
        example_id=example_id,
        question=_normalize(str(ex.get("query") or ex.get("question") or "")),
        answer=answer,
        answer_aliases=(),
        dependencies=tuple(dependencies),
        hop_count=len(dependencies),
    ), audit


def synthetic_dev_corpus() -> tuple[list[CorpusRow], list[EvidenceRow]]:
    """Small non-production corpus for tests and local validation dry-runs."""
    dataset = "devset"
    example_id = "dev_example_001"
    docs = [
        CorpusRow(
            dataset=dataset,
            example_id=example_id,
            corpus_id="dev_doc_alpha",
            document_id="dev_doc_alpha",
            title="Alpha Bridge",
            text="Alpha Bridge says the composer studied in Paris before writing the score.",
            paragraph_id="dev_doc_alpha",
        ),
        CorpusRow(
            dataset=dataset,
            example_id=example_id,
            corpus_id="dev_doc_beta",
            document_id="dev_doc_beta",
            title="Beta Answer",
            text="Beta Answer states that the score was written for the town festival.",
            paragraph_id="dev_doc_beta",
        ),
        CorpusRow(
            dataset=dataset,
            example_id=example_id,
            corpus_id="dev_doc_noise",
            document_id="dev_doc_noise",
            title="Noise",
            text="This unrelated passage discusses weather records and local roads.",
            paragraph_id="dev_doc_noise",
        ),
    ]
    evidence = [
        EvidenceRow(
            dataset=dataset,
            example_id=example_id,
            question="What event was the score written for?",
            answer="the town festival",
            answer_aliases=(),
            dependencies=(
                EvidenceDependency(
                    dependency_id="dev_dep_alpha",
                    document_id="dev_doc_alpha",
                    title="Alpha Bridge",
                    hop_index=0,
                    decomposition_step=0,
                ),
                EvidenceDependency(
                    dependency_id="dev_dep_beta",
                    document_id="dev_doc_beta",
                    title="Beta Answer",
                    hop_index=1,
                    decomposition_step=1,
                    answer_bearing_spans=("the town festival",),
                ),
            ),
            hop_count=2,
        )
    ]
    return docs, evidence


def corpus_hash(rows: Iterable[CorpusRow]) -> str:
    return stable_hash(
        [
            {
                **row.to_json(),
                "text_hash": hash_text(row.text, length=32),
                "text": "",
            }
            for row in rows
        ],
        length=32,
    )


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _paragraph_id(paragraph: Mapping[str, object], fallback_index: int) -> str:
    for key in ("idx", "paragraph_id", "id"):
        if key in paragraph and paragraph.get(key) is not None:
            return str(paragraph.get(key))
    return str(fallback_index)
