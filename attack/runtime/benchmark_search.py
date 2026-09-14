"""Deterministic local BM25 search tool for ODR.

The LangGraph still sees an ordinary ODR search tool. The backend never calls
Tavily, DDG, OpenAI, Anthropic, or any other remote service.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Iterable, Literal, Mapping

from attack.runtime.cues import suffix_for_condition
from attack.runtime.evidence import EvidenceDependency, evaluate_coverage
from attack.runtime.hashing import (
    content_addressed_path,
    hash_text,
    normalize_query,
    stable_hash,
)

try:
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import InjectedToolArg, tool
except Exception:  # pragma: no cover - local unit tests may not install ODR deps.
    RunnableConfig = dict
    InjectedToolArg = object

    def tool(*tool_args, **tool_kwargs):
        def decorate(fn):
            fn.name = tool_args[0] if tool_args and isinstance(tool_args[0], str) else fn.__name__
            fn.metadata = {}
            return fn

        if tool_args and callable(tool_args[0]):
            return decorate(tool_args[0])
        return decorate


BENCHMARK_LOCAL_SEARCH_DESCRIPTION = (
    "Deterministic local benchmark retrieval over approved candidate passages. "
    "Useful for answering benchmark questions without live web access."
)

FORBIDDEN_CORPUS_KEYS = {
    "answer",
    "gold_answer",
    "supporting",
    "is_supporting",
    "supporting_facts",
    "gold_support",
    "required_dependency",
    "evidence_complete",
}

_INDEX_CACHE: dict[str, "BM25Index"] = {}
_RUN_SEARCH_COUNTS: dict[tuple[str, str], int] = defaultdict(int)
_RUN_INJECTION_COUNTS: dict[str, int] = defaultdict(int)


@dataclass(frozen=True)
class BenchmarkDocument:
    dataset: str
    example_id: str
    corpus_id: str
    document_id: str
    title: str
    text: str
    paragraph_id: str = ""

    @classmethod
    def from_json(cls, row: Mapping[str, object]) -> "BenchmarkDocument":
        forbidden = sorted(FORBIDDEN_CORPUS_KEYS.intersection(row))
        if forbidden:
            raise ValueError(f"corpus row leaks evaluation-only keys: {forbidden}")
        return cls(
            dataset=str(row["dataset"]),
            example_id=str(row["example_id"]),
            corpus_id=str(row.get("corpus_id") or row["document_id"]),
            document_id=str(row["document_id"]),
            title=str(row.get("title") or ""),
            text=str(row["text"]),
            paragraph_id=str(row.get("paragraph_id") or row["document_id"]),
        )


@dataclass(frozen=True)
class BenchmarkSearchResult:
    dataset: str
    example_id: str
    corpus_id: str
    document_id: str
    title: str
    text: str
    retrieval_score: float
    rank: int
    query: str
    retriever_config_hash: str
    index_hash: str
    passage_hash: str
    paragraph_id: str = ""

    def metadata(self) -> dict:
        row = asdict(self)
        row.pop("text")
        return row


class BM25Index:
    def __init__(
        self,
        documents: Iterable[BenchmarkDocument],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents = tuple(sorted(documents, key=lambda d: d.document_id))
        self.k1 = float(k1)
        self.b = float(b)
        self.doc_tokens = [_tokenize(doc.title + " " + doc.text) for doc in self.documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_freqs = Counter()
        for tokens in self.doc_tokens:
            self.doc_freqs.update(set(tokens))
        self.index_hash = stable_hash(
            [
                {
                    "dataset": doc.dataset,
                    "example_id": doc.example_id,
                    "corpus_id": doc.corpus_id,
                    "document_id": doc.document_id,
                    "paragraph_id": doc.paragraph_id,
                    "title": doc.title,
                    "text_hash": hash_text(doc.text, length=None),
                }
                for doc in self.documents
            ],
            length=32,
        )
        self.config_hash = stable_hash(
            {"algorithm": "bm25", "k1": self.k1, "b": self.b, "tokenizer": "lower_alnum_v1"},
            length=32,
        )

    def search(self, query: str, top_k: int) -> list[BenchmarkSearchResult]:
        query_terms = _tokenize(query)
        scored = []
        for idx, doc in enumerate(self.documents):
            score = self._score(query_terms, idx)
            scored.append((score, doc.document_id, doc))
        scored.sort(key=lambda item: (-item[0], item[1]))
        results = []
        for rank, (score, _doc_id, doc) in enumerate(scored[:top_k], start=1):
            results.append(
                BenchmarkSearchResult(
                    dataset=doc.dataset,
                    example_id=doc.example_id,
                    corpus_id=doc.corpus_id,
                    document_id=doc.document_id,
                    title=doc.title,
                    text=doc.text,
                    retrieval_score=round(float(score), 8),
                    rank=rank,
                    query=query,
                    retriever_config_hash=self.config_hash,
                    index_hash=self.index_hash,
                    passage_hash=hash_text(doc.text, length=32),
                    paragraph_id=doc.paragraph_id,
                )
            )
        return results

    def _score(self, query_terms: list[str], doc_index: int) -> float:
        if not query_terms or not self.documents:
            return 0.0
        score = 0.0
        doc_len = self.doc_lengths[doc_index]
        freqs = self.term_freqs[doc_index]
        n_docs = len(self.documents)
        for term in query_terms:
            tf = freqs.get(term, 0)
            if tf == 0:
                continue
            df = self.doc_freqs.get(term, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / (self.avg_doc_len or 1.0))
            score += idf * (tf * (self.k1 + 1)) / denom
        return score


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def load_documents(path: Path, *, dataset: str | None = None, example_id: str | None = None) -> list[BenchmarkDocument]:
    docs = []
    with path.open() as fh:
        for raw in fh:
            if not raw.strip():
                continue
            doc = BenchmarkDocument.from_json(json.loads(raw))
            if dataset is not None and doc.dataset != dataset:
                continue
            if example_id is not None and doc.example_id != example_id:
                continue
            docs.append(doc)
    return docs


def get_index(path: Path, *, dataset: str | None, example_id: str | None, retrieval_scope: str) -> BM25Index:
    scope_example_id = example_id if retrieval_scope == "per_example" else None
    key = stable_hash(
        {
            "path": str(path.resolve()),
            "mtime_ns": path.stat().st_mtime_ns,
            "dataset": dataset,
            "example_id": scope_example_id,
            "retrieval_scope": retrieval_scope,
        },
        length=32,
    )
    if key not in _INDEX_CACHE:
        docs = load_documents(path, dataset=dataset, example_id=scope_example_id)
        _INDEX_CACHE[key] = BM25Index(docs)
    return _INDEX_CACHE[key]


def run_benchmark_search(
    queries: list[str],
    *,
    max_results: int = 5,
    config: Mapping[str, object] | None = None,
) -> str:
    cfg = _configurable(config)
    corpus_path = cfg.get("benchmark_corpus_path")
    if not corpus_path:
        return "No benchmark_local corpus configured. Set benchmark_corpus_path in RunnableConfig.configurable."

    dataset = _optional_str(cfg.get("benchmark_dataset") or cfg.get("v3_dataset"))
    example_id = _optional_str(cfg.get("benchmark_example_id") or cfg.get("v3_example_id"))
    retrieval_scope = str(cfg.get("benchmark_retrieval_scope") or "per_example")
    if retrieval_scope not in {"per_example", "dataset"}:
        raise ValueError(f"unsupported benchmark_retrieval_scope: {retrieval_scope}")
    if retrieval_scope == "per_example" and not example_id:
        raise ValueError("per_example benchmark retrieval requires benchmark_example_id or v3_example_id")

    top_k = int(cfg.get("benchmark_top_k") or max_results)
    condition = str(cfg.get("v3_condition") or "NaturalClean")
    cue_overrides = cfg.get("v3_cue_overrides") if isinstance(cfg.get("v3_cue_overrides"), Mapping) else {}
    cue = suffix_for_condition(condition, cue_overrides)
    run_id = str(cfg.get("v3_run_id") or cfg.get("thread_id") or "default")
    tool_context = cfg.get("v3_tool_context") if isinstance(cfg.get("v3_tool_context"), Mapping) else {}

    search_call_index = _next_search_call_index(run_id, str(tool_context.get("researcher_subgraph_id") or "default"))
    is_target = _is_target_tool_message(cfg, tool_context, search_call_index)
    oracle_mode = cue.oracle_complete and is_target
    evidence_dependencies = _load_evidence_dependencies(
        Path(str(cfg["benchmark_evidence_path"]))
        if cfg.get("benchmark_evidence_path")
        else None,
        dataset,
        example_id,
    )

    query_results = []
    oracle_info: dict[str, object] = {}
    for query in queries:
        if oracle_mode:
            results, oracle_info = _oracle_complete_results(Path(str(corpus_path)), cfg, query)
        else:
            index = get_index(
                Path(str(corpus_path)),
                dataset=dataset,
                example_id=example_id,
                retrieval_scope=retrieval_scope,
            )
            results = index.search(query, top_k)
        query_results.extend(results)

    cache_dir = Path(str(cfg.get("benchmark_cache_dir") or "results/cache"))
    _write_content_cache(cache_dir, query_results)

    base_output = format_results(query_results)
    if oracle_mode and not oracle_info.get("oracle_complete_available", False):
        base_output = _oracle_unavailable_message(oracle_info)
    coverage = evaluate_coverage(
        evidence_dependencies,
        [result.document_id for result in query_results],
        retrieved_paragraph_ids=[result.paragraph_id for result in query_results],
    )
    suffix = (
        cue.suffix
        if cue.requires_injection
        and is_target
        and _RUN_INJECTION_COUNTS[run_id] == 0
        and not (oracle_mode and not oracle_info.get("oracle_complete_available", False))
        else ""
    )
    attacked = bool(suffix)
    if attacked:
        _RUN_INJECTION_COUNTS[run_id] += 1
    output = base_output + (f"\n\n{suffix}" if suffix else "")

    event = {
        "event_type": "benchmark_search",
        "run_id": run_id,
        "condition": condition,
        "dataset": dataset,
        "example_id": example_id,
        "queries": queries,
        "normalized_queries": [normalize_query(query) for query in queries],
        "search_query_hash": stable_hash([normalize_query(query) for query in queries], length=32),
        "top_k": top_k,
        "retrieval_scope": retrieval_scope,
        "oracle_mode": oracle_mode,
        **oracle_info,
        "supervisor_delegation_index": tool_context.get("supervisor_delegation_index"),
        "supervisor_tool_call_id": tool_context.get("supervisor_tool_call_id"),
        "researcher_subgraph_id": tool_context.get("researcher_subgraph_id"),
        "search_call_index_within_researcher": search_call_index,
        "search_tool_call_id": tool_context.get("tool_call_id"),
        "search_tool_message_id": tool_context.get("tool_call_id"),
        "attacked_observation": attacked,
        "suffix_hash": cue.suffix_hash if attacked else "",
        "hop_count": len(evidence_dependencies),
        "required_dependency_count": len(evidence_dependencies),
        "required_paragraph_ids": coverage.required_paragraph_ids,
        "required_dependency_ids": coverage.required_dependency_ids,
        "retrieved_paragraph_ids_after_observation": coverage.retrieved_paragraph_ids,
        "observed_document_ids_after_observation": coverage.observed_document_ids,
        "covered_dependency_ids_after_observation": coverage.covered_dependency_ids,
        "dependencies_seen_after_observation": coverage.satisfied_dependency_ids,
        "missing_dependency_ids_after_observation": coverage.remaining_dependency_ids,
        "coverage_status_after_observation": coverage.status,
        "coverage_fraction_after_observation": coverage.coverage_fraction,
        "dependencies_seen_at_injection": coverage.satisfied_dependency_ids if is_target else (),
        "missing_dependency_ids_at_injection": coverage.remaining_dependency_ids if is_target else (),
        "coverage_fraction_at_injection": coverage.coverage_fraction if is_target else None,
        "coverage_before_cue": coverage.to_json() if is_target else None,
        "base_tool_message_hash": hash_text(base_output, length=32),
        "tool_message_hash": hash_text(output, length=32),
        "shared_prefix_hash": stable_hash(
            {
                "dataset": dataset,
                "example_id": example_id,
                "queries": [normalize_query(query) for query in queries],
                "base_tool_message_hash": hash_text(base_output, length=32),
                "supervisor_delegation_index": tool_context.get("supervisor_delegation_index"),
                "search_call_index_within_researcher": search_call_index,
            },
            length=32,
        ),
        "results": [result.metadata() for result in query_results],
    }
    _write_search_cache(cache_dir, event)
    if trace_path := cfg.get("benchmark_trace_path"):
        _append_jsonl(Path(str(trace_path)), event)

    return output


@tool("tavily_search", description=BENCHMARK_LOCAL_SEARCH_DESCRIPTION)
async def benchmark_search(
    queries: list[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    config: RunnableConfig = None,
) -> str:
    del topic
    return await asyncio.to_thread(run_benchmark_search, queries, max_results=max_results, config=config)


def format_results(results: list[BenchmarkSearchResult]) -> str:
    if not results:
        return "No benchmark_local search results found."
    lines = ["Search results from benchmark_local deterministic retrieval:"]
    for result in results:
        lines.extend(
            [
                "",
                f"--- SOURCE {result.rank}: {result.title} ---",
                f"URL: benchmark://{result.dataset}/{result.example_id}/{result.document_id}",
                f"DATASET: {result.dataset}",
                f"EXAMPLE_ID: {result.example_id}",
                f"DOCUMENT_ID: {result.document_id}",
                f"RETRIEVAL_SCORE: {result.retrieval_score:.8f}",
                f"RANK: {result.rank}",
                f"QUERY: {result.query}",
                "",
                "SUMMARY:",
                result.text,
                "",
                "-" * 80,
            ]
        )
    return "\n".join(lines)


def _configurable(config: Mapping[str, object] | None) -> Mapping[str, object]:
    if not config:
        return {}
    configurable = config.get("configurable") if isinstance(config, Mapping) else None
    return configurable if isinstance(configurable, Mapping) else config


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _next_search_call_index(run_id: str, researcher_subgraph_id: str) -> int:
    key = (run_id, researcher_subgraph_id)
    current = _RUN_SEARCH_COUNTS[key]
    _RUN_SEARCH_COUNTS[key] += 1
    return current


def _is_target_tool_message(cfg: Mapping[str, object], tool_context: Mapping[str, object], search_call_index: int) -> bool:
    target_delegation = int(cfg.get("v3_target_supervisor_delegation_index") or 0)
    target_search_call = int(cfg.get("v3_target_search_call_index") or 0)
    delegation = tool_context.get("supervisor_delegation_index")
    if delegation is None:
        delegation = 0
    tool_name = str(tool_context.get("tool_name") or "tavily_search")
    return (
        int(delegation) == target_delegation
        and search_call_index == target_search_call
        and tool_name in {"tavily_search", "benchmark_search", "web_search"}
    )


def _write_content_cache(cache_dir: Path, results: list[BenchmarkSearchResult]) -> None:
    for result in results:
        path = content_addressed_path(cache_dir, "passages", result.passage_hash, ".txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(result.text, encoding="utf-8")


def _write_search_cache(cache_dir: Path, event: Mapping[str, object]) -> None:
    key = stable_hash(
        {
            "dataset": event.get("dataset"),
            "example_id": event.get("example_id"),
            "search_query_hash": event.get("search_query_hash"),
            "top_k": event.get("top_k"),
            "retrieval_scope": event.get("retrieval_scope"),
            "result_hashes": [
                result.get("passage_hash")
                for result in event.get("results", [])
                if isinstance(result, Mapping)
            ],
        },
        length=32,
    )
    path = content_addressed_path(cache_dir, "search", key, ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(event, sort_keys=True, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def _oracle_complete_results(
    corpus_path: Path,
    cfg: Mapping[str, object],
    query: str,
) -> tuple[list[BenchmarkSearchResult], dict[str, object]]:
    evidence_path = cfg.get("benchmark_evidence_path")
    dataset = _optional_str(cfg.get("benchmark_dataset") or cfg.get("v3_dataset"))
    example_id = _optional_str(cfg.get("benchmark_example_id") or cfg.get("v3_example_id"))
    if not evidence_path or not dataset or not example_id:
        return [], {
            "oracle_complete_available": False,
            "oracle_unavailable_reason": "missing evidence path, dataset, or example id",
        }

    dependencies = _load_evidence_dependencies(Path(str(evidence_path)), dataset, example_id)
    required_doc_ids = _unique(dep.document_id for dep in dependencies if not dep.ambiguous)
    required_paragraph_ids = _unique(
        dep.supporting_paragraph_id or dep.document_id
        for dep in dependencies
        if not dep.ambiguous
    )
    if not dependencies:
        return [], {
            "oracle_complete_available": False,
            "oracle_unavailable_reason": "no dependency annotations available",
            "oracle_required_dependency_count": 0,
            "oracle_supplied_dependency_count": 0,
            "oracle_required_paragraph_count": 0,
            "oracle_supplied_paragraph_count": 0,
            "oracle_all_dependency_annotations_available": False,
        }
    if any(dep.ambiguous for dep in dependencies):
        return [], {
            "oracle_complete_available": False,
            "oracle_unavailable_reason": "ambiguous dependency annotations",
            "oracle_required_dependency_count": len(dependencies),
            "oracle_supplied_dependency_count": 0,
            "oracle_required_paragraph_count": len(required_paragraph_ids),
            "oracle_supplied_paragraph_count": 0,
            "oracle_all_dependency_annotations_available": False,
        }
    docs = [
        doc
        for doc in load_documents(corpus_path, dataset=dataset, example_id=example_id)
        if doc.document_id in required_doc_ids
    ]
    supplied_doc_ids = {doc.document_id for doc in docs}
    missing_doc_ids = [doc_id for doc_id in required_doc_ids if doc_id not in supplied_doc_ids]
    if missing_doc_ids:
        return [], {
            "oracle_complete_available": False,
            "oracle_unavailable_reason": "required support documents missing from corpus",
            "oracle_missing_document_ids": missing_doc_ids,
            "oracle_required_dependency_count": len(dependencies),
            "oracle_supplied_dependency_count": 0,
            "oracle_required_paragraph_count": len(required_paragraph_ids),
            "oracle_supplied_paragraph_count": len(docs),
            "oracle_all_dependency_annotations_available": False,
        }
    docs.sort(key=lambda doc: required_doc_ids.index(doc.document_id))
    index = BM25Index(docs)
    results = []
    for rank, doc in enumerate(docs, start=1):
        results.append(
            BenchmarkSearchResult(
                dataset=doc.dataset,
                example_id=doc.example_id,
                corpus_id=doc.corpus_id,
                document_id=doc.document_id,
                title=doc.title,
                text=doc.text,
                retrieval_score=1.0,
                rank=rank,
                query=query,
                retriever_config_hash=index.config_hash,
                index_hash=index.index_hash,
                passage_hash=hash_text(doc.text, length=32),
                paragraph_id=doc.paragraph_id,
            )
        )
    coverage = evaluate_coverage(
        dependencies,
        [result.document_id for result in results],
        retrieved_paragraph_ids=[result.paragraph_id for result in results],
    )
    return results, {
        "oracle_complete_available": coverage.is_complete,
        "oracle_unavailable_reason": "" if coverage.is_complete else "oracle result did not cover every dependency",
        "oracle_required_dependency_count": len(dependencies),
        "oracle_supplied_dependency_count": len(coverage.covered_dependency_ids),
        "oracle_required_paragraph_count": len(required_paragraph_ids),
        "oracle_supplied_paragraph_count": len(coverage.retrieved_paragraph_ids),
        "oracle_all_dependency_annotations_available": coverage.is_complete,
        "oracle_coverage_before_cue": coverage.to_json(),
    }


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _required_doc_ids(evidence_path: Path, dataset: str, example_id: str) -> list[str]:
    required_doc_ids = []
    with evidence_path.open() as fh:
        for raw in fh:
            if not raw.strip():
                continue
            row = json.loads(raw)
            if str(row.get("dataset")) != dataset or str(row.get("example_id")) != example_id:
                continue
            for dep in row.get("dependencies", []):
                doc_id = dep.get("document_id")
                if doc_id is not None and str(doc_id) not in required_doc_ids:
                    required_doc_ids.append(str(doc_id))
    return required_doc_ids


def _load_evidence_dependencies(
    evidence_path: Path | None,
    dataset: str | None,
    example_id: str | None,
) -> tuple[EvidenceDependency, ...]:
    if not evidence_path or not dataset or not example_id or not evidence_path.exists():
        return ()
    with evidence_path.open() as fh:
        for raw in fh:
            if not raw.strip():
                continue
            row = json.loads(raw)
            if str(row.get("dataset")) != dataset or str(row.get("example_id")) != example_id:
                continue
            return tuple(_dependency_from_json(dep) for dep in row.get("dependencies", []))
    return ()


def _dependency_from_json(dep: Mapping[str, object]) -> EvidenceDependency:
    return EvidenceDependency(
        dependency_id=str(dep["dependency_id"]),
        document_id=str(dep["document_id"]),
        title=str(dep.get("title") or ""),
        hop_index=dep.get("hop_index"),
        supporting_paragraph_id=str(dep.get("supporting_paragraph_id") or ""),
        sentence_ids=tuple(dep.get("sentence_ids") or ()),
        supporting_spans=tuple(dep.get("supporting_spans") or ()),
        decomposition_question=str(dep.get("decomposition_question") or ""),
        intermediate_answer=str(dep.get("intermediate_answer") or ""),
        bridge_entities=tuple(dep.get("bridge_entities") or ()),
        required_relations=tuple(dep.get("required_relations") or ()),
        answer_bearing_spans=tuple(dep.get("answer_bearing_spans") or ()),
        decomposition_step=dep.get("decomposition_step"),
        ambiguous=bool(dep.get("ambiguous") or False),
    )


def _oracle_unavailable_message(oracle_info: Mapping[str, object]) -> str:
    reason = oracle_info.get("oracle_unavailable_reason") or "unknown reason"
    return f"Oracle-complete condition unavailable for this example: {reason}."
