"""Evidence dependency and coverage utilities.

Coverage is computed from retrieval metadata, never from suffix text or final
answer text. This is what keeps cue wording from counting as evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Mapping


EVIDENCE_NONE = "none"
EVIDENCE_RELEVANT_INCOMPLETE = "relevant_incomplete"
EVIDENCE_COMPLETE = "complete"
EVIDENCE_AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class EvidenceDependency:
    dependency_id: str
    document_id: str
    title: str = ""
    hop_index: int | None = None
    supporting_paragraph_id: str = ""
    sentence_ids: tuple[int, ...] = ()
    supporting_spans: tuple[str, ...] = ()
    decomposition_question: str = ""
    intermediate_answer: str = ""
    bridge_entities: tuple[str, ...] = ()
    required_relations: tuple[str, ...] = ()
    answer_bearing_spans: tuple[str, ...] = ()
    decomposition_step: int | None = None
    ambiguous: bool = False

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceCoverage:
    status: str
    required_paragraph_ids: tuple[str, ...]
    required_dependency_ids: tuple[str, ...]
    retrieved_paragraph_ids: tuple[str, ...]
    observed_document_ids: tuple[str, ...]
    covered_dependency_ids: tuple[str, ...]
    satisfied_dependency_ids: tuple[str, ...]
    remaining_dependency_ids: tuple[str, ...]
    ambiguous_dependency_ids: tuple[str, ...] = field(default_factory=tuple)
    coverage_fraction: float = 0.0

    @property
    def is_complete(self) -> bool:
        return self.status == EVIDENCE_COMPLETE

    @property
    def is_relevant_incomplete(self) -> bool:
        return self.status == EVIDENCE_RELEVANT_INCOMPLETE

    def to_json(self) -> dict:
        return asdict(self)


def coverage_from_retrieval_metadata(
    dependencies: Iterable[EvidenceDependency],
    result_metadata: Iterable[Mapping[str, object]],
) -> EvidenceCoverage:
    """Compute coverage using only retrieved ids from search metadata."""
    observed_doc_ids = []
    retrieved_paragraph_ids = []
    for row in result_metadata:
        doc_id = row.get("document_id") or row.get("doc_id")
        if doc_id is not None:
            observed_doc_ids.append(str(doc_id))
        paragraph_id = row.get("paragraph_id") or row.get("supporting_paragraph_id") or doc_id
        if paragraph_id is not None:
            retrieved_paragraph_ids.append(str(paragraph_id))
    return evaluate_coverage(dependencies, observed_doc_ids, retrieved_paragraph_ids=retrieved_paragraph_ids)


def evaluate_coverage(
    dependencies: Iterable[EvidenceDependency],
    observed_document_ids: Iterable[str],
    *,
    retrieved_paragraph_ids: Iterable[str] | None = None,
) -> EvidenceCoverage:
    deps = tuple(dependencies)
    observed_input = tuple(str(doc_id) for doc_id in observed_document_ids)
    observed = tuple(dict.fromkeys(observed_input))
    retrieved_paragraphs = tuple(
        dict.fromkeys(
            str(paragraph_id)
            for paragraph_id in (
                retrieved_paragraph_ids if retrieved_paragraph_ids is not None else observed_input
            )
        )
    )
    observed_set = set(observed)
    paragraph_set = set(retrieved_paragraphs)
    required_dependency_ids = tuple(dep.dependency_id for dep in deps)
    required_paragraph_ids = tuple(
        dict.fromkeys(_dependency_paragraph_id(dep) for dep in deps if _dependency_paragraph_id(dep))
    )

    if not deps:
        return EvidenceCoverage(
            status=EVIDENCE_AMBIGUOUS,
            required_paragraph_ids=(),
            required_dependency_ids=(),
            retrieved_paragraph_ids=retrieved_paragraphs,
            observed_document_ids=observed,
            covered_dependency_ids=(),
            satisfied_dependency_ids=(),
            remaining_dependency_ids=(),
            coverage_fraction=0.0,
        )

    ambiguous = tuple(dep.dependency_id for dep in deps if dep.ambiguous)
    if ambiguous:
        return EvidenceCoverage(
            status=EVIDENCE_AMBIGUOUS,
            required_paragraph_ids=required_paragraph_ids,
            required_dependency_ids=required_dependency_ids,
            retrieved_paragraph_ids=retrieved_paragraphs,
            observed_document_ids=observed,
            covered_dependency_ids=(),
            satisfied_dependency_ids=(),
            remaining_dependency_ids=required_dependency_ids,
            ambiguous_dependency_ids=ambiguous,
            coverage_fraction=0.0,
        )

    satisfied = tuple(
        dep.dependency_id
        for dep in deps
        if dep.document_id in observed_set or _dependency_paragraph_id(dep) in paragraph_set
    )
    satisfied_set = set(satisfied)
    remaining = tuple(dep.dependency_id for dep in deps if dep.dependency_id not in satisfied_set)
    coverage_fraction = len(satisfied) / len(deps)

    if len(satisfied) == len(deps):
        status = EVIDENCE_COMPLETE
    elif satisfied:
        status = EVIDENCE_RELEVANT_INCOMPLETE
    else:
        status = EVIDENCE_NONE

    return EvidenceCoverage(
        status=status,
        required_paragraph_ids=required_paragraph_ids,
        required_dependency_ids=required_dependency_ids,
        retrieved_paragraph_ids=retrieved_paragraphs,
        observed_document_ids=observed,
        covered_dependency_ids=satisfied,
        satisfied_dependency_ids=satisfied,
        remaining_dependency_ids=remaining,
        ambiguous_dependency_ids=(),
        coverage_fraction=coverage_fraction,
    )


def first_full_coverage_step(
    dependencies: Iterable[EvidenceDependency],
    observed_document_ids_by_step: Iterable[Iterable[str]],
) -> int | None:
    """Return the first zero-based step with complete evidence coverage."""
    cumulative = []
    for step, observed_document_ids in enumerate(observed_document_ids_by_step):
        cumulative.extend(str(doc_id) for doc_id in observed_document_ids)
        if evaluate_coverage(dependencies, cumulative).is_complete:
            return step
    return None


def _dependency_paragraph_id(dep: EvidenceDependency) -> str:
    return dep.supporting_paragraph_id or dep.document_id
