"""Unsupported-bridge arm: same relational wording, relation not supported by d1.

Construction (pre-specified; nothing depends on victim outcomes):
  1. Distractor: the first candidate entity, in a fixed order, that is
     (a) a different entity from both arguments of a*,
     (b) judged the same kind of entity as a*'s tail (attacker, temperature 0),
     (c) such that the swapped statement f' (a*'s statement with the tail
         replaced by the distractor) is entailed by NO sentence of d1 under
         the same NLI check used for grounding.
     Candidate order: tails of other extracted facts with a*'s relation (by
     span offset); then same-kind entities proposed by the attacker
     (temperature 0) that occur VERBATIM in a d1 document's text, by first
     occurrence (added after dev showed the fact/title pool rarely holds a
     same-type entity, e.g. no second date); then heads/tails of all other
     extracted facts (by span offset, unlocated last); then document titles
     (by position). At most MAX_CANDIDATES are tried.
  2. Minimal pair: for EVERY variant in the bridge-sufficiency K pool, its
     counterpart is the identical text with each literal occurrence of the
     tail replaced by the distractor. A variant whose text does not contain
     the tail literally has no counterpart.
  3. Each counterpart runs every validity gate; V4 (no invented task facts)
     is recorded but exempt, because the unsupported relation is the point.
At selection time the arm takes the counterpart of the u* chosen for
bridge_sufficiency_k3, so the two arms differ only in the swapped entity.
"""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any, Mapping

from prematurestop.pipelines.bridge_mechanism import config as C
from prematurestop.cues.dynamic.bridge import fill, overlap, read_prompt
from prematurestop.cues.dynamic.generation import STATUS_OK, Variant, evaluate_variant
from prematurestop.cues.dynamic.entail import Entailer
from prematurestop.cues.dynamic.llm import ChatClient
from prematurestop.cues.dynamic.observation import documents_for_prompt, locate_span, parse_observation

STATUS_NO_DISTRACTOR = "NO_DISTRACTOR"
STATUS_NO_COUNTERPART = "NO_COUNTERPART"
MAX_CANDIDATES = 8
TYPE_SCHEMA = {"type": "object", "properties": {"same_type": {"type": "boolean"}}, "required": ["same_type"]}
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _same_entity(a: str, b: str) -> bool:
    return overlap(a, b) >= C.ENTITY_SLOT_MIN_OVERLAP or overlap(b, a) >= C.ENTITY_SLOT_MIN_OVERLAP


CANDIDATES_SCHEMA = {"type": "object", "properties": {"candidates": {"type": "array", "items": {"type": "string"}}},
                     "required": ["candidates"]}
MAX_PROPOSED = 8


def proposed_same_kind(client: ChatClient, observation: str, target: str) -> list[str]:
    """Attacker-proposed same-kind entities, kept only if verbatim in d1 text, by first offset."""
    docs = parse_observation(observation)
    user = fill(read_prompt("entity_candidates_v1.user.txt"),
                {"DOCUMENTS": documents_for_prompt(docs), "TARGET": target, "N": str(MAX_PROPOSED)})
    msgs = [{"role": "system", "content": read_prompt("entity_candidates_v1.system.txt")}, {"role": "user", "content": user}]
    res = client.complete_json(msgs, schema=CANDIDATES_SCHEMA, schema_name="entity_candidates",
                               temperature=C.JUDGE_TEMPERATURE, top_p=1.0, max_tokens=200, seed=C.JUDGE_SEED)
    raw = (res.parsed or {}).get("candidates", []) if isinstance(res.parsed, Mapping) else []
    located = []
    for cand in raw[:MAX_PROPOSED]:
        cand = str(cand).strip()
        loc = locate_span(observation, docs, cand) if cand else None
        if loc is not None:
            located.append((loc[0], cand))
    return [c for _, c in sorted(located)]


def distractor_candidates(bridge_rec: Mapping[str, Any], proposed: list[str] | None = None) -> list[str]:
    a = bridge_rec["anchor"]
    facts = bridge_rec.get("facts", [])
    key = lambda f: (f.get("span_offset") is None, f.get("span_offset") or 0)
    ordered: list[str] = []
    ordered += [f["tail"] for f in sorted(facts, key=key) if f.get("relation") == a["relation"]]
    ordered += list(proposed or [])
    for f in sorted(facts, key=key):
        ordered += [f.get("tail", ""), f.get("head", "")]
    ordered += [d["title"] for d in sorted(bridge_rec.get("documents", []), key=lambda d: d["position"])]
    out, seen = [], set()
    for c in ordered:
        c = (c or "").strip()
        if not c or c.lower() in seen or c.startswith("?"):
            continue
        if _same_entity(c, a["bridge_entity_tail"]) or _same_entity(c, a["bridge_entity_head"]):
            continue
        seen.add(c.lower())
        out.append(c)
    return out[:MAX_CANDIDATES]


def swapped_statement(anchor: Mapping[str, Any], distractor: str) -> str:
    stmt, tail = anchor["bridge_fact"], anchor["bridge_entity_tail"]
    if tail and tail in stmt:
        return stmt.replace(tail, distractor)
    return f"{anchor['bridge_entity_head']} {anchor['relation']} {distractor}."


def counterpart_text(cue: str, tail: str, distractor: str) -> str | None:
    if not cue or not tail:
        return None
    if tail in cue:
        return cue.replace(tail, distractor)
    pat = re.compile(re.escape(tail), re.IGNORECASE)
    return pat.sub(distractor, cue) if pat.search(cue) else None


def choose_distractor(client: ChatClient, entailer: Entailer, bridge_rec: Mapping[str, Any],
                      observation: str) -> dict:
    a = bridge_rec["anchor"]
    docs = parse_observation(observation)
    docs_text = documents_for_prompt(docs)
    sentences = [s for d in docs for s in _SENT_RE.split(d.text) if s.strip()]
    proposed = proposed_same_kind(client, observation, a["bridge_entity_tail"])
    tried = []
    for cand in distractor_candidates(bridge_rec, proposed):
        user = fill(read_prompt("entity_type_v1.user.txt"),
                    {"DOCUMENTS": docs_text, "CANDIDATE": cand, "TARGET": a["bridge_entity_tail"]})
        msgs = [{"role": "system", "content": read_prompt("entity_type_v1.system.txt")}, {"role": "user", "content": user}]
        res = client.complete_json(msgs, schema=TYPE_SCHEMA, schema_name="entity_type",
                                   temperature=C.JUDGE_TEMPERATURE, top_p=1.0, max_tokens=20, seed=C.JUDGE_SEED)
        same = bool(isinstance(res.parsed, Mapping) and res.parsed.get("same_type") is True)
        entry = {"candidate": cand, "same_type": same}
        if same:
            stmt = swapped_statement(a, cand)
            probs = [entailer.entails(premise=s, hypothesis=stmt) for s in sentences]
            supported = any(p.entails for p in probs)
            entry.update(swapped_statement=stmt, supported_by_d1=supported,
                         max_entail_prob=max((p.entail_prob for p in probs), default=0.0))
            tried.append(entry)
            if not supported:
                return {"status": STATUS_OK, "distractor": cand, "swapped_statement": stmt, "tried": tried,
                        "proposed": proposed}
        else:
            tried.append(entry)
    return {"status": STATUS_NO_DISTRACTOR, "distractor": None, "tried": tried, "proposed": proposed}


def build_unsupported(*, client: ChatClient, entailer: Entailer, token_counter, question: str, observation: str,
                      bridge_rec: Mapping[str, Any], bridge_cue_rec: Mapping[str, Any],
                      generation_config: Mapping[str, Any]) -> dict:
    """Counterparts for every variant of the bridge-sufficiency pool (index-aligned)."""
    a = bridge_rec["anchor"]
    dist = choose_distractor(client, entailer, bridge_rec, observation)
    out: dict[str, Any] = {"condition": C.CUE_1, "mode": "unsupported", "distractor": dist, "variants": []}
    if dist["status"] != STATUS_OK:
        out.update(status=STATUS_NO_DISTRACTOR, n_valid=0)
        return out
    docs = parse_observation(observation)
    docs_text = documents_for_prompt(docs)
    obs_text = "\n".join(d.title + "\n" + d.text for d in docs)
    for src in bridge_cue_rec.get("variants", []):
        text = counterpart_text(src.get("cue", ""), a["bridge_entity_tail"], dist["distractor"])
        v = Variant(index=src["index"], seed=src["seed"], raw="", cue=text or "")
        if text is None:
            v.errors = {"COUNTERPART": ["tail_not_literally_in_source_variant"]}
            v.checks = {k: False for k in ("V1", "V2", "V3", "V4", "V5", "COPY", "ENTITY")}
        else:
            v = evaluate_variant(v, client=client, token_counter=token_counter, question=question,
                                 observation_text=obs_text, docs_text=docs_text,
                                 answer_slot=generation_config.get("answer_slot", ""), condition=C.CUE_1,
                                 anchor=a, gold_answers=generation_config.get("gold_answers", ()),
                                 mode="bridge", exempt=("V4",))
        d = asdict(v)
        d["source_variant_valid"] = bool(src.get("valid"))
        out["variants"].append(d)
    n_valid = sum(1 for v in out["variants"] if v["valid"])
    out.update(status=STATUS_OK if n_valid else STATUS_NO_COUNTERPART, n_valid=n_valid,
               n_variants=len(out["variants"]))
    return out
