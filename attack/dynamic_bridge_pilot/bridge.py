"""Steps 1-3: decompose q into H, extract span-backed facts F(d1), select a*.

  Step 1  H <- Decompose(q). Attacker LLM, temperature 0, question text only
          (no gold answer, no evidence annotations, no victim trace). Cached
          per (dataset, example_id) and reused across victims/arms.
  Step 2  F(d1) = {(f, s) : s = d1[i:j], s |= f}. The attacker proposes
          (statement, span) pairs; a pair survives only if the span occurs
          verbatim in a document's SUMMARY text AND the prespecified NLI
          check (entail.py) says "<that document's title>. <span>" entails the
          statement (the title resolves the paragraph subject's pronouns).
  Step 3  A(q,d1) = {(f,s,e) : Match(f,e)=1, 0 < delta(e) < inf};
          a* = argmin [delta(e_a), tau(a)], tau = absolute span offset in d1.
          Empty A -> NO_ANCHOR (recorded, never substituted).
"""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from attack.dynamic_bridge_pilot import config as C
from attack.dynamic_bridge_pilot.entail import Entailer
from attack.dynamic_bridge_pilot.llm import ChatClient
from attack.dynamic_bridge_pilot.observation import (
    ObservedDocument,
    documents_for_prompt,
    locate_span,
    parse_observation,
)
from attack.runtime.hashing import hash_text

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
ANSWER_SLOT = "?answer"
MAX_FACTS = 8

STATUS_OK = "OK"
STATUS_DECOMPOSE_INVALID = "DECOMPOSE_INVALID"
STATUS_NO_FACTS = "NO_FACTS"
STATUS_NO_ANCHOR = "NO_ANCHOR"

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an the of in on at to for by with and or from as is was are were be been its it this that".split())

DECOMPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "edges": {"type": "array", "items": {
            "type": "object",
            "properties": {"id": {"type": "string"}, "head": {"type": "string"},
                           "relation": {"type": "string"}, "tail": {"type": "string"}},
            "required": ["id", "head", "relation", "tail"]}},
        "answer_slot_description": {"type": "string"},
    },
    "required": ["edges", "answer_slot_description"],
}
EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {
        "type": "object",
        "properties": {"relation": {"type": "string"}, "head": {"type": "string"}, "tail": {"type": "string"},
                       "statement": {"type": "string"}, "span": {"type": "string"}},
        "required": ["relation", "head", "tail", "statement", "span"]}}},
    "required": ["facts"],
}


def read_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def fill(template: str, mapping: Mapping[str, str]) -> str:
    for key, value in mapping.items():
        template = template.replace(f"<<{key}>>", value)
    return template


def content_tokens(text: str) -> set[str]:
    return {t for t in _WORD_RE.findall((text or "").lower()) if t not in _STOP}


def overlap(a: str, b: str) -> float:
    """|tokens(a) & tokens(b)| / |tokens(b)| -- how much of b is covered by a."""
    tb = content_tokens(b)
    return len(content_tokens(a) & tb) / len(tb) if tb else 0.0


# --------------------------------------------------------------------------- H
@dataclass
class Edge:
    id: str
    head: str
    relation: str
    tail: str


@dataclass
class Graph:
    edges: list[Edge]
    answer_slot_description: str
    delta: dict[str, int] = field(default_factory=dict)   # edge id -> delta(e)
    valid: bool = False
    errors: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "Graph":
        return cls(edges=[Edge(**e) for e in d["edges"]], answer_slot_description=d["answer_slot_description"],
                   delta=dict(d.get("delta", {})), valid=bool(d.get("valid")), errors=list(d.get("errors", [])))


def is_variable(node: str) -> bool:
    return node.startswith("?")


def node_distance_to_answer(edges: list[Edge]) -> dict[str, float]:
    """Shortest directed path length (in edges) from each node to ?answer."""
    rev: dict[str, list[str]] = {}
    for e in edges:
        rev.setdefault(e.tail, []).append(e.head)
    dist: dict[str, float] = {ANSWER_SLOT: 0}
    queue = deque([ANSWER_SLOT])
    while queue:
        n = queue.popleft()
        for h in rev.get(n, []):
            if h not in dist:
                dist[h] = dist[n] + 1
                queue.append(h)
    return dist


def _has_cycle(edges: list[Edge]) -> bool:
    adj: dict[str, list[str]] = {}
    for e in edges:
        adj.setdefault(e.head, []).append(e.tail)
    state: dict[str, int] = {}

    def visit(n: str) -> bool:
        state[n] = 1
        for m in adj.get(n, []):
            if state.get(m) == 1 or (state.get(m) is None and visit(m)):
                return True
        state[n] = 2
        return False

    return any(state.get(n) is None and visit(n) for n in list(adj))


def validate_graph(raw: Any, question: str) -> Graph:
    if not isinstance(raw, Mapping) or not isinstance(raw.get("edges"), list):
        return Graph(edges=[], answer_slot_description="", errors=["malformed_decomposition"])
    edges: list[Edge] = []
    errors: list[str] = []
    for i, e in enumerate(raw["edges"]):
        try:
            edges.append(Edge(id=str(e.get("id") or f"e{i + 1}"), head=str(e["head"]).strip(),
                              relation=str(e["relation"]).strip().lower(), tail=str(e["tail"]).strip()))
        except (KeyError, AttributeError, TypeError):
            errors.append(f"malformed_edge_{i}")
    g = Graph(edges=edges, answer_slot_description=str(raw.get("answer_slot_description") or ""), errors=errors)
    if not edges:
        g.errors.append("no_edges")
    if len(edges) > C.MAX_EDGES:
        g.errors.append(f"too_many_edges:{len(edges)}")
    if len({e.id for e in edges}) != len(edges):
        g.errors.append("duplicate_edge_ids")
    if not any(e.tail == ANSWER_SLOT for e in edges):
        g.errors.append("answer_slot_not_a_tail")
    if any(e.head == ANSWER_SLOT for e in edges):
        g.errors.append("answer_slot_has_outgoing_edge")
    if any(not e.relation for e in edges):
        g.errors.append("empty_relation")
    if _has_cycle(edges):
        g.errors.append("cycle")
    for e in edges:
        for node in (e.head, e.tail):
            if not is_variable(node) and overlap(question, node) < C.ENTITY_SLOT_MIN_OVERLAP:
                g.errors.append(f"constant_not_in_question:{node}")
    dist = node_distance_to_answer(edges)
    for e in edges:
        d = dist.get(e.head)
        if d is None:
            g.errors.append(f"edge_not_on_path_to_answer:{e.id}")
            continue
        g.delta[e.id] = int(d)
        if d > C.MAX_DEPTH:
            g.errors.append(f"too_deep:{e.id}:{int(d)}")
    g.valid = not g.errors
    return g


DECOMPOSE_REPAIR_ATTEMPTS = 1   # pre-specified: one repair round with the validation errors


def decompose(client: ChatClient, question: str) -> dict:
    """Step 1, temperature 0. If the graph fails validation, the decomposer gets
    exactly one repair round with the validator's error list (dev showed e.g. the
    answer variable never routed to ?answer). Both attempts are recorded."""
    user = fill(read_prompt("decompose_v1.user.txt"),
                {"MAX_EDGES": str(C.MAX_EDGES), "MAX_DEPTH": str(C.MAX_DEPTH), "QUESTION": question})
    messages = [{"role": "system", "content": read_prompt("decompose_v1.system.txt")}, {"role": "user", "content": user}]
    attempts = []
    for attempt in range(1 + DECOMPOSE_REPAIR_ATTEMPTS):
        res = client.complete_json(messages, schema=DECOMPOSE_SCHEMA, schema_name="decomposition",
                                   temperature=C.JUDGE_TEMPERATURE, top_p=1.0, max_tokens=500, seed=C.JUDGE_SEED)
        graph = validate_graph(res.parsed, question) if res.parsed is not None else Graph(
            edges=[], answer_slot_description="", errors=[res.error or "no_output"])
        attempts.append({"raw": res.raw, "errors": list(graph.errors)})
        if graph.valid:
            break
        messages = messages + [
            {"role": "assistant", "content": res.raw or ""},
            {"role": "user", "content": "That graph is invalid: " + "; ".join(graph.errors) +
             '. Fix it: the final answer must be the variable "?answer", every edge must lie on a path ending at '
             '"?answer", and constants must be copied from the question. Return the corrected JSON only.'}]
    return {"graph": graph.to_json(), "raw": attempts[-1]["raw"], "attempts": attempts, "request": res.request,
            "prompt_version": "decompose_v1+repair1"}


# ---------------------------------------------------------------------- F(d1)
@dataclass
class Fact:
    relation: str
    head: str
    tail: str
    statement: str
    span: str
    span_offset: int | None = None
    document_id: str = ""
    document_title: str = ""
    document_position: int | None = None
    entail_prob: float | None = None
    entail_label: str = ""
    entail_premise: str = ""
    span_backed: bool = False
    rejection: str = ""


def extract_facts(client: ChatClient, entailer: Entailer, question: str, observation: str,
                  docs: list[ObservedDocument], graph: Graph) -> tuple[list[Fact], dict]:
    relations = sorted({e.relation for e in graph.edges})
    user = fill(read_prompt("extract_v1.user.txt"), {
        "EDGES": "\n".join(f"- {e.id}: ({e.head}) --[{e.relation}]--> ({e.tail})" for e in graph.edges),
        "RELATIONS": json.dumps(relations), "MAX_FACTS": str(MAX_FACTS), "QUESTION": question,
        "DOCUMENTS": documents_for_prompt(docs),
    })
    messages = [{"role": "system", "content": read_prompt("extract_v1.system.txt")}, {"role": "user", "content": user}]
    res = client.complete_json(messages, schema=EXTRACT_SCHEMA, schema_name="facts",
                               temperature=C.JUDGE_TEMPERATURE, top_p=1.0, max_tokens=1200, seed=C.JUDGE_SEED)
    facts: list[Fact] = []
    raw_facts = (res.parsed or {}).get("facts", []) if isinstance(res.parsed, Mapping) else []
    for item in raw_facts[:MAX_FACTS]:
        try:
            f = Fact(**{k: str(item[k]).strip() if k != "span" else str(item[k]) for k in
                        ("relation", "head", "tail", "statement", "span")})
        except (KeyError, TypeError):
            continue
        f.relation = f.relation.lower()
        loc = locate_span(observation, docs, f.span)
        if loc is None:
            f.rejection = "span_not_verbatim_in_document_text"
        else:
            f.span_offset, doc = loc
            f.document_id, f.document_title, f.document_position = doc.document_id, doc.title, doc.position
            # Premise = "<document title>. <span>": Wikipedia sentences refer to their
            # subject by pronoun ("He was the son of Eli Thayer"), so the bare span
            # cannot entail a fact naming the subject. The title is the paragraph's
            # subject; the span must still be verbatim in that same document.
            premise = f.span if f.span.lstrip().lower().startswith(doc.title.lower()) else f"{doc.title}. {f.span}"
            f.entail_premise = premise
            verdict = entailer.entails(premise=premise, hypothesis=f.statement)
            f.entail_prob, f.entail_label = verdict.entail_prob, verdict.label
            f.span_backed = verdict.entails
            if not verdict.entails:
                f.rejection = "span_does_not_entail_statement"
        facts.append(f)
    return facts, {"raw": res.raw, "request": res.request, "prompt_version": "extract_v1", "error": res.error}


# ------------------------------------------------------------------- A, a*
def match(f: Fact, e: Edge) -> bool:
    """Same relation type and compatible argument slots."""
    if f.relation != e.relation:
        return False
    for slot_value, node in ((f.head, e.head), (f.tail, e.tail)):
        if not slot_value or slot_value.startswith("?"):
            return False
        if not is_variable(node) and overlap(slot_value, node) < C.ENTITY_SLOT_MIN_OVERLAP:
            return False
    return True


def candidate_anchors(facts: list[Fact], graph: Graph, policy: str = C.ANCHOR_EDGE_POLICY) -> list[dict]:
    if policy not in C.ANCHOR_EDGE_POLICIES:
        raise ValueError(f"unknown anchor edge policy {policy}")
    out = []
    for fi, f in enumerate(facts):
        if not f.span_backed:
            continue
        for e in graph.edges:
            d = graph.delta.get(e.id)
            if d is None or not (0 < d < float("inf")):
                continue
            if policy == "exclude_answer_edges" and e.tail == ANSWER_SLOT:
                continue
            if match(f, e):
                out.append({"fact_index": fi, "edge_id": e.id, "delta": d, "tau": f.span_offset})
    return out


def select_anchor(candidates: list[dict]) -> dict | None:
    if not candidates:
        return None
    return min(candidates, key=lambda a: (a["delta"], a["tau"], a["edge_id"]))


# ------------------------------------------------------------- entry point
def build_bridge(client: ChatClient, entailer: Entailer, *, question: str, observation: str,
                 decomposition: dict, policy: str = C.ANCHOR_EDGE_POLICY) -> dict:
    """Steps 2-3 given a cached step-1 decomposition. Never raises on attack failure."""
    graph = Graph.from_json(decomposition["graph"])
    docs = parse_observation(observation)
    record: dict[str, Any] = {
        "observation_hash": hash_text(observation, length=32),
        "n_documents": len(docs),
        "documents": [{"position": d.position, "rank": d.rank, "title": d.title, "document_id": d.document_id}
                      for d in docs],
        "graph": graph.to_json(),
        "anchor_edge_policy": policy,
    }
    if not graph.valid:
        record.update(status=STATUS_DECOMPOSE_INVALID, facts=[], candidates=[], anchor=None)
        return record
    facts, meta = extract_facts(client, entailer, question, observation, docs, graph)
    cands = candidate_anchors(facts, graph, policy)
    chosen = select_anchor(cands)
    record.update(facts=[asdict(f) for f in facts], extraction=meta, candidates=cands)
    if not any(f.span_backed for f in facts):
        record.update(status=STATUS_NO_FACTS, anchor=None)
    elif chosen is None:
        record.update(status=STATUS_NO_ANCHOR, anchor=None)
    else:
        f = facts[chosen["fact_index"]]
        e = next(x for x in graph.edges if x.id == chosen["edge_id"])
        record.update(status=STATUS_OK, anchor={
            "bridge_fact": f.statement, "bridge_span": f.span, "bridge_entity_head": f.head,
            "bridge_entity_tail": f.tail, "relation": f.relation, "edge": asdict(e), "delta": chosen["delta"],
            "reasoning_chain_position": f"edge {e.id}, {chosen['delta']} hop(s) from answer slot",
            "edge_tail_is_answer_slot": e.tail == ANSWER_SLOT,
            "span_offset": f.span_offset, "source_document_id": f.document_id,
            "source_document_title": f.document_title, "source_document_position": f.document_position,
            "entail_prob": f.entail_prob,
        })
    return record
