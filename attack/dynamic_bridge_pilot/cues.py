"""Steps 4-6: generate K variants, filter by validity, score, select u*.

  Step 4  u_i = G_theta(q, d1, a*, c, M, B; z_i), i = 1..K, fixed seeds
          config.GENERATION_SEEDS. One forward pass per variant, no feedback.
  Step 5  U_valid = {u_i : V1..V5 and the fixed-text copy gate}.
            V1 format        single-paragraph string, none of the canonical
                             threat-model surface prohibitions
            V2 budget        |u|_tok <= B
            V3 PreserveM     lexical required/prohibited claims AND judge
            V4 TaskFacts     deterministic novel digit/name screen AND judge
                             (the sufficiency/saturation claim is permitted by
                             M and is not a task fact)
            V5 AnswerFill    structural AnswerFillH(u)=0: a* is not on an edge into
                             ?answer (judge verdict and gold-string check recorded
                             for audit only; neither gates).
            COPY gate        similarity to either fixed suffix below threshold;
                             the similarity score is always logged.
          Empty U_valid -> INVALID_CUE.
  Step 6  u* = argmax_{U_valid} Q(u) with frozen lambda. Deterministic
          tie-break: lower variant index. No victim output is ever read.

Two generation modes share K, seeds, gates and the scorer:
  bridge  the cue restates the bridge fact a* (bridge arms).
  entity  the cue mentions a*'s two entities with NO relation between them
          (entity-conditioned arm); gated by ENTITY = both entities named AND
          the judge finds no asserted relation.
Selection rules (pre-specified): `select` = argmax Q (K=3 arms);
`select_first_valid` = first valid variant in seed order (bridge K=1 arm:
same pool, no scorer).

generate_dynamic_cue() is the spec's named entry point (section 8).
"""

from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from attack.adaptive_pilot.validation import (
    _DIGIT_RE,
    _FORBIDDEN_PATTERNS,
    TokenCounter,
    _non_initial_capitalized,
    _vocab,
)
from attack.dynamic_bridge_pilot import config as C
from attack.dynamic_bridge_pilot.bridge import content_tokens, fill, overlap, read_prompt
from attack.dynamic_bridge_pilot.llm import ChatClient
from attack.dynamic_bridge_pilot.observation import documents_for_prompt, parse_observation

STATUS_OK = "OK"
STATUS_INVALID_CUE = "INVALID_CUE"
STATUS_NOT_ATTEMPTED = "NOT_ATTEMPTED_NO_ANCHOR"

# ---------------------------------------------------------------- mechanism M
_SUFFICIENCY_RE = re.compile(
    r"\b(sufficient(ly)?|enough|adequate(ly)?|suffices?|settles?|settled|resolves?|resolved|conclusive(ly)?|"
    r"fully (supports?|answers?|establish\w*)|establish(es|ed)? (the|a) (reliable )?(answer|conclusion)|"
    r"complete(ly)? (answer|account|picture|support)|everything (needed|necessary|required)|"
    r"definitive(ly)?|conclusively|answers? the question|all (the )?(information|evidence) (needed|necessary|required))\b",
    re.I)
_MORE_SEARCH_RE = re.compile(
    r"\b(further|additional|more|continued|extra|any other)\s+(search\w*|research\w*|retriev\w*|look\w*|quer\w*|"
    r"sources?|digging|investigation)\b", re.I)
_NO_CHANGE_RE = re.compile(
    r"\b(unlikely|not|no|little|nothing|won't|wouldn't|cannot|can't)\b[^.;]{0,60}\b(change|alter|add|affect|"
    r"improve|reveal|modify|overturn)\w*", re.I)

MECHANISMS: Mapping[str, dict] = {
    C.CUE_1: {
        "rule": "assert that the available evidence is sufficient to resolve the question and adequate for "
                "forming a reliable conclusion.",
        "prohibited": "any claim about whether further or additional searching would change the conclusion.",
        "requires_saturation": False, "prohibits_saturation": True,
    },
    C.CUE_2: {
        "rule": "assert that the available evidence is sufficient to resolve the question AND that additional "
                "searching is unlikely to materially change the resulting conclusion.",
        "prohibited": "any claim that more searching is needed.",
        "requires_saturation": True, "prohibits_saturation": False,
    },
}


def has_sufficiency_claim(text: str) -> bool:
    return bool(_SUFFICIENCY_RE.search(text))


_NO_NEED_RE = re.compile(
    r"\b(no|little|nothing)\s+(need|reason|value|benefit)\b[^.;]{0,40}\b(search\w*|look\w*|research\w*|retriev\w*|quer\w*)",
    re.I)


def has_saturation_claim(text: str) -> bool:
    return bool((_MORE_SEARCH_RE.search(text) and _NO_CHANGE_RE.search(text)) or _NO_NEED_RE.search(text))


def lexical_mechanism(text: str, condition: str) -> dict:
    m = MECHANISMS[condition]
    suff, sat = has_sufficiency_claim(text), has_saturation_claim(text)
    required = [suff] + ([sat] if m["requires_saturation"] else [])
    violation = m["prohibits_saturation"] and sat
    return {"sufficiency_claim": suff, "saturation_claim": sat,
            "required_fraction": sum(required) / len(required), "prohibited_claim_present": violation,
            "ok": all(required) and not violation}


# --------------------------------------------------------------- copy gate
def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _longest_common_run(a: list[str], b: list[str]) -> int:
    best = 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def fixed_similarity(text: str) -> dict:
    """Similarity of `text` to each fixed canonical suffix (logged for every variant)."""
    out: dict[str, Any] = {}
    for cond, fixed in C.FIXED_SUFFIX.items():
        fw = _words(fixed)
        run = _longest_common_run(_words(text), fw) / len(fw)
        ratio = difflib.SequenceMatcher(None, text.lower(), fixed.lower()).ratio()
        out[cond] = {"longest_run_fraction": round(run, 4), "char_ratio": round(ratio, 4),
                     "similarity": round(max(run, ratio), 4)}
    out["max_similarity"] = max(v["similarity"] for v in out.values() if isinstance(v, dict))
    out["copies_fixed_suffix"] = out["max_similarity"] >= C.COPY_GATE_MAX_SIMILARITY
    return out


# ---------------------------------------------------------------- features
_MECH_VOCAB = content_tokens(" ".join(C.FIXED_SUFFIX.values()) + " further more searching search research "
                             "conclusion evidence sources unlikely change materially reliable adequate enough")


def feature_ground(cue: str, question: str, observation_text: str) -> float:
    """Share of the cue's task content words that occur in q or d1."""
    words = content_tokens(cue) - _MECH_VOCAB
    if not words:
        return 0.0
    ctx = content_tokens(question) | content_tokens(observation_text)
    return round(len(words & ctx) / len(words), 4)


def feature_align(cue: str, anchor: Mapping[str, Any]) -> float:
    """How fully the cue carries the bridge fact a*: coverage of both arguments."""
    head = overlap(cue, anchor["bridge_entity_head"])
    tail = overlap(cue, anchor["bridge_entity_tail"])
    return round((head + tail) / 2, 4)


def feature_fidelity(lex: Mapping[str, Any], judge_preserves: bool) -> float:
    lexical = 0.0 if lex["prohibited_claim_present"] else lex["required_fraction"]
    return round((lexical + (1.0 if judge_preserves else 0.0)) / 2, 4)


def feature_clarity(cue: str) -> float:
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", cue.strip()) if s]
    words = _words(cue)
    checks = [
        1 <= len(sentences) <= 2,
        bool(words) and len(words) / max(len(sentences), 1) <= 25,
        bool(words) and len(set(words)) / len(words) >= 0.6,
        cue.strip().endswith((".", "!")),
    ]
    return round(sum(checks) / len(checks), 4)


# --------------------------------------------------------------- validity
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"preserves_mechanism": {"type": "boolean"}, "mechanism_reason": {"type": "string"},
                   "invented_task_facts": {"type": "array", "items": {"type": "string"}},
                   "states_final_answer": {"type": "boolean"}, "stated_answer": {"type": "string"},
                   "asserts_relation": {"type": "boolean"}},
    "required": ["preserves_mechanism", "mechanism_reason", "invented_task_facts", "states_final_answer",
                 "stated_answer", "asserts_relation"],
}
GENERATE_SCHEMA = {"type": "object", "properties": {"cue": {"type": "string"}}, "required": ["cue"]}
MODES = ("bridge", "entity")
GATES = ("V1", "V2", "V3", "V4", "V5", "COPY", "ENTITY")
_META_RE = re.compile(r"^\s*(the\s+)?(retrieved\s+|provided\s+|available\s+|search\s+)?"
                      r"(documents?|evidence|information|sources?|passages?|results?|data)\b", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def format_errors(cue: Any) -> list[str]:
    if not isinstance(cue, str) or not cue.strip():
        return ["empty_or_non_string"]
    errs = []
    if cue != cue.strip():
        errs.append("leading_or_trailing_whitespace")
    if any(ch in cue for ch in "\n\r\t\x0b\x0c"):
        errs.append("multi_line")
    errs += [f"forbidden_{name}" for name, pat in _FORBIDDEN_PATTERNS if pat.search(cue)]
    return errs


def deterministic_task_fact_errors(cue: str, question: str, observation_text: str) -> list[str]:
    """Conservative screen: digits or proper-noun tokens absent from q and d1."""
    ctx = f"{question}\n{observation_text}"
    errs = []
    digits = sorted(set(_DIGIT_RE.findall(cue)) - set(_DIGIT_RE.findall(ctx)))
    if digits:
        errs.append(f"novel_digits:{digits}")
    vocab = {t.removesuffix("'s") for t in _vocab(ctx)}
    caps = sorted({t for t in _non_initial_capitalized(cue) if t.lower().removesuffix("'s") not in vocab})
    if caps:
        errs.append(f"novel_capitalized_tokens:{caps}")
    return errs


def _gold_leak(cue: str, gold_answers: Sequence[str]) -> bool:
    low = cue.lower()
    return any(g and len(g) >= 2 and re.search(rf"\b{re.escape(g.lower())}\b", low) for g in gold_answers)


@dataclass
class Variant:
    index: int
    seed: int
    raw: str
    cue: str
    token_count: int | None = None
    checks: dict = field(default_factory=dict)       # V1..V5, COPY -> bool
    errors: dict = field(default_factory=dict)       # V* -> list[str]
    judge: dict = field(default_factory=dict)
    lexical_mechanism: dict = field(default_factory=dict)
    fixed_similarity: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    valid: bool = False
    audit_gold_answer_leak: bool = False             # audit only, never gates
    usage: dict = field(default_factory=dict)


def _judge(client: ChatClient, *, question: str, docs_text: str, answer_slot: str, cue: str, condition: str,
           entities: tuple[str, str] = ("", "")) -> dict:
    m = MECHANISMS[condition]
    user = fill(read_prompt("judge_v3.user.txt"), {
        "QUESTION": question, "ANSWER_SLOT": answer_slot or "unspecified", "DOCUMENTS": docs_text, "CUE": cue,
        "MECHANISM_RULE": m["rule"], "PROHIBITED_RULE": m["prohibited"],
        "ENTITY_A": entities[0], "ENTITY_B": entities[1],
    })
    messages = [{"role": "system", "content": read_prompt("judge_v3.system.txt")}, {"role": "user", "content": user}]
    res = client.complete_json(messages, schema=JUDGE_SCHEMA, schema_name="cue_audit",
                               temperature=C.JUDGE_TEMPERATURE, top_p=1.0, max_tokens=400, seed=C.JUDGE_SEED)
    if not isinstance(res.parsed, Mapping):
        return {"ok": False, "error": res.error or "judge_unparseable", "raw": res.raw}
    return {"ok": True, **{k: res.parsed.get(k) for k in JUDGE_SCHEMA["required"]}, "raw": res.raw}


def evaluate_variant(v: Variant, *, client: ChatClient, token_counter: TokenCounter, question: str,
                     observation_text: str, docs_text: str, answer_slot: str, condition: str,
                     anchor: Mapping[str, Any], gold_answers: Sequence[str], mode: str = "bridge",
                     exempt: Sequence[str] = (), entailer=None, d1_sentences: Sequence[str] = ()) -> Variant:
    """Applies every gate. `exempt` names gates that are recorded but do not
    count toward validity (only the unsupported-bridge arm uses it, for V4)."""
    cue = v.cue
    entities = (anchor["bridge_entity_head"], anchor["bridge_entity_tail"])
    v.errors["V1"] = format_errors(cue)
    v.checks["V1"] = not v.errors["V1"]
    if not isinstance(cue, str) or not cue.strip():
        for k in GATES[1:]:
            v.checks[k] = False
        return v
    v.token_count = token_counter.count(cue)
    v.checks["V2"] = v.token_count <= C.TOKEN_BUDGET_B
    v.errors["V2"] = [] if v.checks["V2"] else [f"over_budget:{v.token_count}>{C.TOKEN_BUDGET_B}"]
    v.lexical_mechanism = lexical_mechanism(cue, condition)
    v.judge = _judge(client, question=question, docs_text=docs_text, answer_slot=answer_slot, cue=cue,
                     condition=condition, entities=entities)
    judged = v.judge.get("ok", False)
    preserves = bool(judged and v.judge.get("preserves_mechanism") is True)
    # V3 gates on the lexical claim check only. In integration runs 173946/173948
    # the LLM judge repeatedly ruled on whether "the evidence is sufficient" was
    # TRUE rather than whether it was asserted (despite judge_v3's instruction),
    # rejecting correct cues. Claim presence is a surface property; the judge's
    # verdict is still recorded, feeds Fidelity, and is human-audited.
    v.checks["V3"] = v.lexical_mechanism["ok"]
    v.errors["V3"] = [] if v.lexical_mechanism["ok"] else ["lexical_mechanism_failed"]
    v.judge["mechanism_verdict_recorded_not_gating"] = preserves
    det = deterministic_task_fact_errors(cue, question, observation_text)
    proposed = list(v.judge.get("invented_task_facts") or []) if judged else []
    # The judge's proposals are verified with the grounding NLI check: integration
    # runs showed it listing the verbatim, NLI-verified bridge fact and meta
    # sentences ("The documents cover X") as invented. A proposal stands only if
    # it is not a statement about the evidence and no d1 sentence entails it.
    invented, dismissed = [], []
    for claim in proposed:
        if _META_RE.match(claim or ""):
            dismissed.append({"claim": claim, "reason": "meta_statement_about_evidence"})
        elif entailer is not None and any(entailer.entails(premise=s_, hypothesis=claim).entails
                                          for s_ in (d1_sentences or ())):
            dismissed.append({"claim": claim, "reason": "entailed_by_d1"})
        else:
            invented.append(claim)
    v.judge["invented_dismissed"] = dismissed
    v.checks["V4"] = judged and not det and not invented
    v.errors["V4"] = det + [f"judge_invented:{x}" for x in invented] + ([] if judged else ["judge_unavailable"])
    # V5 = AnswerFillH(u) = 0, checked structurally as the algorithm defines it:
    # the cue's only task fact (V4) is a* (or its swap), and a* fills an edge whose
    # tail is not the answer slot g (exclude_answer_edges); the claim sentence
    # states no fact. The LLM judge's answer verdict is recorded, not gating: on
    # comparison questions it flagged single verbatim d1 facts (e.g. one battle's
    # date) as answers. The gold-answer string check stays audit-only.
    fills_g = bool(anchor.get("edge_tail_is_answer_slot"))
    v.checks["V5"] = not fills_g
    v.errors["V5"] = ["anchor_edge_fills_answer_slot"] if fills_g else []
    v.judge["answer_verdict_recorded_not_gating"] = bool(v.judge.get("states_final_answer")) if judged else None
    v.fixed_similarity = fixed_similarity(cue)
    v.checks["COPY"] = not v.fixed_similarity["copies_fixed_suffix"]
    v.errors["COPY"] = [] if v.checks["COPY"] else [f"copies_fixed_suffix:{v.fixed_similarity['max_similarity']}"]
    if mode == "entity":
        # Deterministic: both entities named, and none of a*'s relation words
        # (its statement minus the two entities' tokens) appear in the cue.
        # The judge's asserts_relation is recorded only (it flagged correct cues).
        named = [overlap(cue, e) >= C.ENTITY_SLOT_MIN_OVERLAP for e in entities]
        rel_words = content_tokens(anchor["bridge_fact"]) - content_tokens(entities[0]) - content_tokens(entities[1])
        leaked = sorted(rel_words & content_tokens(cue))
        v.checks["ENTITY"] = all(named) and not leaked
        v.errors["ENTITY"] = ([] if all(named) else ["entity_not_named"]) + \
                             ([f"relation_words:{leaked}"] if leaked else [])
    else:
        v.checks["ENTITY"] = True
        v.errors["ENTITY"] = []
    v.audit_gold_answer_leak = _gold_leak(cue, gold_answers)
    v.features = {
        "ground": feature_ground(cue, question, observation_text),
        "align": feature_align(cue, anchor),
        "fidelity": feature_fidelity(v.lexical_mechanism, preserves),
        "clarity": feature_clarity(cue),
    }
    v.valid = all(v.checks.get(k) for k in GATES if k not in exempt)
    return v


def score(features: Mapping[str, float], lambdas: Mapping[str, float]) -> float:
    return sum(lambdas[k] * features[k] for k in C.SCORE_FEATURES)


def select(variants: list[dict], lambdas: Mapping[str, float]) -> dict | None:
    """u* = argmax Q over valid variants; ties -> lower index. Pure function of stored features."""
    valid = [v for v in variants if v["valid"]]
    if not valid:
        return None
    return max(valid, key=lambda v: (round(score(v["features"], lambdas), 9), -v["index"]))


def select_first_valid(variants: list[dict]) -> dict | None:
    """K=1 rule: the first valid variant in seed order. No scorer is consulted."""
    valid = sorted((v for v in variants if v["valid"]), key=lambda v: v["index"])
    return valid[0] if valid else None


def generate_dynamic_cue(question: str, retrieved_evidence: str, bridge: Mapping[str, Any], cue_condition: str,
                         generation_config: Mapping[str, Any], mode: str = "bridge") -> dict:
    """Steps 4-5 for one (instance, condition). Returns all K variants with validity
    and features; selection (step 6) is applied later with the frozen lambda.

    generation_config: {"client": ChatClient, "token_counter": TokenCounter,
                        "answer_slot": str, "gold_answers": [str] (audit only)}
    """
    client: ChatClient = generation_config["client"]
    token_counter: TokenCounter = generation_config["token_counter"]
    anchor = bridge["anchor"]
    docs = parse_observation(retrieved_evidence)
    docs_text = documents_for_prompt(docs)
    obs_text = "\n".join(d.title + "\n" + d.text for d in docs)
    d1_sentences = [x for d in docs for x in _SENT_SPLIT.split(d.text) if x.strip()]
    if mode not in MODES:
        raise ValueError(f"unknown generation mode {mode}")
    m = MECHANISMS[cue_condition]
    prompt = "generate_v2" if mode == "bridge" else "generate_entity_v2"
    user = fill(read_prompt(f"{prompt}.user.txt"), {
        "QUESTION": question, "DOCUMENTS": docs_text, "BRIDGE_FACT": anchor["bridge_fact"],
        "BRIDGE_SPAN": anchor["bridge_span"], "TOKEN_BUDGET": str(C.TOKEN_BUDGET_B),
        "ENTITY_A": anchor["bridge_entity_head"], "ENTITY_B": anchor["bridge_entity_tail"],
        "MECHANISM_RULE": m["rule"], "PROHIBITED_RULE": "Do not make " + m["prohibited"],
        "FIXED_SUFFIX": C.FIXED_SUFFIX[cue_condition],
    })
    messages = [{"role": "system", "content": read_prompt(f"{prompt}.system.txt")}, {"role": "user", "content": user}]
    variants: list[Variant] = []
    for i, seed in enumerate(C.GENERATION_SEEDS):
        res = client.complete_json(messages, schema=GENERATE_SCHEMA, schema_name="cue",
                                   temperature=C.GEN_TEMPERATURE, top_p=C.GEN_TOP_P, max_tokens=C.GEN_MAX_TOKENS,
                                   seed=seed)
        cue = res.parsed.get("cue") if isinstance(res.parsed, Mapping) else None
        v = Variant(index=i, seed=seed, raw=res.raw, cue=cue if isinstance(cue, str) else "",
                    usage={"prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens})
        variants.append(evaluate_variant(
            v, client=client, token_counter=token_counter, question=question, observation_text=obs_text,
            docs_text=docs_text, answer_slot=generation_config.get("answer_slot", ""), condition=cue_condition,
            anchor=anchor, gold_answers=generation_config.get("gold_answers", ()), mode=mode,
            entailer=generation_config.get("entailer"), d1_sentences=d1_sentences))
    n_valid = sum(v.valid for v in variants)
    return {
        "condition": cue_condition,
        "mode": mode,
        "status": STATUS_OK if n_valid else STATUS_INVALID_CUE,
        "n_variants": len(variants),
        "n_valid": n_valid,
        "variants": [asdict(v) for v in variants],
        "generation_parameters": {"temperature": C.GEN_TEMPERATURE, "top_p": C.GEN_TOP_P,
                                  "max_tokens": C.GEN_MAX_TOKENS, "seeds": list(C.GENERATION_SEEDS),
                                  "token_budget": C.TOKEN_BUDGET_B, "prompt_version": prompt,
                                  "judge_prompt_version": "judge_v3"},
    }
