"""Condition-blind final-answer evaluator for production outputs.

The extractor in this module receives only the benchmark question and the raw
ODR final answer. Gold answers and experimental condition names are used only
after extraction, in the semantic scoring pass.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import re
import string
from typing import Any, Iterable, Mapping, Sequence


EVALUATOR_VERSION = "answer_evaluator_v2_rule_based_20260903"

EXTRACTION_SPEC = """Condition-blind extraction rule:
Given only a benchmark question and a raw final answer, extract the concise
answer span the model intended as its answer. Do not use gold labels, aliases,
model names, dataset names, or experimental condition names. Preserve the raw
answer separately. If the answer is missing, refused, malformed, or presents
multiple unresolved candidates, return short_answer=null with the corresponding
failure reason."""

SEMANTIC_SPEC = """Semantic scoring rule:
After condition-blind extraction, compare the extracted answer with the question,
gold answer, and official aliases. Mark correct for exact matches, accepted
aliases, obvious semantic equivalents, yes/no equivalents, complete list answers,
and answer-bearing occurrences inside longer sentences. Mark incorrect for
partials, contradictions, negated gold mentions, incidental mentions, missing
answers, and refusals. Mark needs_adjudication for unresolved conflicts or
high-overlap cases without a safe deterministic decision."""


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_ANSWER_LINE_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:final\s+answer|answer|short\s+answer)\s*:\s*(.+?)\s*$",
    re.IGNORECASE,
)
_MONTH_RE = (
    r"January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?"
)
_MONTH_DATE_RE = re.compile(rf"\b(?:{_MONTH_RE})\s+\d{{1,2}},\s+\d{{3,4}}\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b")
_SOURCE_MARK_RE = re.compile(r"\s*\[\d+\]\s*")


@dataclass(frozen=True)
class ExtractionResult:
    short_answer: str | None
    extractable: bool
    reason: str


@dataclass(frozen=True)
class SemanticResult:
    semantic_label: str
    semantic_correct: bool
    reason: str
    matched_gold_or_alias: str | None


@dataclass(frozen=True)
class OfficialScores:
    exact_match: float
    token_f1: float
    matched_gold_or_alias: str | None


@dataclass(frozen=True)
class FullEvaluation:
    extraction: ExtractionResult
    semantic: SemanticResult
    official_scores: OfficialScores
    gold_containment: bool
    cache_key: str


def prompt_hashes() -> dict[str, str]:
    return {
        "extraction_spec_sha256": _sha256_text(EXTRACTION_SPEC),
        "semantic_spec_sha256": _sha256_text(SEMANTIC_SPEC),
    }


def cache_key(question: str, raw_final_answer: str, evaluator_version: str = EVALUATOR_VERSION) -> str:
    payload = json.dumps(
        {
            "question": question or "",
            "raw_final_answer": raw_final_answer or "",
            "evaluator_version": evaluator_version,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return _sha256_text(payload)


def normalize_answer(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("'", "")
    text = text.replace("’", "")
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def official_em_f1(prediction: str | None, gold: str) -> tuple[float, float]:
    pred_norm = normalize_answer(prediction or "")
    gold_norm = normalize_answer(gold or "")
    em = 1.0 if pred_norm == gold_norm and gold_norm else 0.0
    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()
    if not pred_tokens or not gold_tokens:
        return em, 1.0 if pred_tokens == gold_tokens and gold_tokens else 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return em, 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return em, 2 * precision * recall / (precision + recall)


def official_scores(prediction: str | None, gold: str, aliases: Sequence[str] | None = None) -> OfficialScores:
    choices = _answer_choices(gold, aliases)
    best_em = 0.0
    best_f1 = 0.0
    best_alias = None
    for choice in choices:
        em, f1 = official_em_f1(prediction, choice)
        if (em, f1) > (best_em, best_f1):
            best_em = em
            best_f1 = f1
            best_alias = choice
    return OfficialScores(exact_match=best_em, token_f1=best_f1, matched_gold_or_alias=best_alias)


def extract_short_answer(question: str, raw_final_answer: str) -> ExtractionResult:
    """Extract a concise answer without gold, aliases, condition, model, or dataset."""

    text = _clean_raw_answer(raw_final_answer)
    if not text:
        return ExtractionResult(short_answer=None, extractable=False, reason="no_answer")

    candidate = _answer_line(text)
    if not candidate and _contains_answer_label(text):
        return ExtractionResult(short_answer=None, extractable=False, reason="malformed_output")
    candidate = candidate or _first_answer_block(text)
    candidate = _strip_leadin(_clean_candidate(candidate))
    if not candidate:
        return ExtractionResult(short_answer=None, extractable=False, reason="malformed_output")
    if _is_refusal(candidate):
        return ExtractionResult(short_answer=None, extractable=False, reason="refusal")
    if _has_unresolved_conflict(candidate):
        return ExtractionResult(short_answer=None, extractable=False, reason="conflicting_answers")

    shortened, reason = _question_guided_shortening(question, candidate)
    shortened = _clean_candidate(shortened)
    if not shortened:
        return ExtractionResult(short_answer=None, extractable=False, reason="malformed_output")
    return ExtractionResult(short_answer=shortened, extractable=True, reason=reason)


def evaluate_answer_v2(
    *,
    question: str,
    raw_final_answer: str,
    gold_answer: str,
    gold_aliases: Sequence[str] | None = None,
) -> FullEvaluation:
    extraction = extract_short_answer(question, raw_final_answer)
    scores = official_scores(extraction.short_answer, gold_answer, gold_aliases)
    semantic = semantic_accuracy(
        question=question,
        extraction=extraction,
        gold_answer=gold_answer,
        gold_aliases=gold_aliases or (),
        scores=scores,
    )
    containment = gold_contained_diagnostic(raw_final_answer, gold_answer, gold_aliases)
    return FullEvaluation(
        extraction=extraction,
        semantic=semantic,
        official_scores=scores,
        gold_containment=containment,
        cache_key=cache_key(question, raw_final_answer),
    )


def semantic_accuracy(
    *,
    question: str,
    extraction: ExtractionResult,
    gold_answer: str,
    gold_aliases: Sequence[str] | None = None,
    scores: OfficialScores | None = None,
) -> SemanticResult:
    del question
    if not extraction.extractable or not extraction.short_answer:
        if extraction.reason == "conflicting_answers":
            return SemanticResult("needs_adjudication", False, "conflicting_candidate_answers", None)
        return SemanticResult("incorrect", False, extraction.reason, None)

    prediction = extraction.short_answer
    pred_norm = normalize_answer(prediction)
    choices = _answer_choices(gold_answer, gold_aliases)
    scored = scores or official_scores(prediction, gold_answer, gold_aliases)

    if _answer_is_refusal_or_missing(prediction):
        return SemanticResult("incorrect", False, "missing_or_refusal_answer", None)
    if _has_unresolved_conflict(prediction):
        return SemanticResult("needs_adjudication", False, "conflicting_candidate_answers", None)
    if scored.exact_match == 1.0:
        return SemanticResult("correct", True, "official_exact_match", scored.matched_gold_or_alias)

    yes_no = _score_yes_no(pred_norm, choices)
    if yes_no is not None:
        label, matched = yes_no
        return SemanticResult(label, label == "correct", "yes_no_semantic_match", matched)

    list_result = _score_list_answer(prediction, choices)
    if list_result is not None:
        label, reason, matched = list_result
        return SemanticResult(label, label == "correct", reason, matched)

    for choice in choices:
        choice_norm = normalize_answer(choice)
        if not choice_norm:
            continue
        if _negated_or_alternative_mention(pred_norm, choice_norm):
            return SemanticResult("incorrect", False, "gold_mentioned_negated_or_as_rejected_alternative", choice)
        if _semantic_equivalent(choice_norm, pred_norm):
            return SemanticResult("correct", True, "semantic_equivalence", choice)
        if _contains_answer_phrase(pred_norm, choice_norm):
            if _multiple_candidate_answer_contains_gold(pred_norm, choice_norm):
                return SemanticResult("needs_adjudication", False, "multiple_candidate_answer_contains_gold", choice)
            return SemanticResult("correct", True, "answer_phrase_in_extracted_answer", choice)

    if scored.token_f1 >= 0.85:
        return SemanticResult("needs_adjudication", False, "high_token_overlap_without_safe_match", scored.matched_gold_or_alias)
    return SemanticResult("incorrect", False, "no_supported_semantic_equivalence", None)


def gold_contained_diagnostic(text: str | None, gold_answer: str, aliases: Sequence[str] | None = None) -> bool:
    haystack = normalize_answer(text or "")
    for choice in _answer_choices(gold_answer, aliases):
        needle = normalize_answer(choice)
        if needle and _contains_answer_phrase(haystack, needle):
            return True
    return False


def summarize_predictions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    assigned = len(rows)
    valid = sum(1 for row in rows if row.get("valid_output"))
    extracted = sum(1 for row in rows if row.get("extractable"))
    correct = sum(1 for row in rows if row.get("semantic_correct"))
    needs = sum(1 for row in rows if row.get("semantic_label") == "needs_adjudication")
    containment = sum(1 for row in rows if row.get("gold_containment"))
    em_sum = sum(float(row.get("exact_match") or 0.0) for row in rows)
    f1_sum = sum(float(row.get("token_f1") or 0.0) for row in rows)
    return {
        "assigned": assigned,
        "valid": valid,
        "extracted": extracted,
        "correct": correct,
        "needs_adjudication": needs,
        "gold_containment": containment,
        "answer_accuracy": correct / assigned if assigned else 0.0,
        "exact_match": em_sum / assigned if assigned else 0.0,
        "token_f1": f1_sum / assigned if assigned else 0.0,
        "gold_containment_rate": containment / assigned if assigned else 0.0,
        "extraction_success_rate": extracted / assigned if assigned else 0.0,
        "needs_adjudication_rate": needs / assigned if assigned else 0.0,
        "valid_output_rate": valid / assigned if assigned else 0.0,
    }


def percentage_point_delta(treatment_rate: float, control_rate: float) -> float:
    return 100.0 * (float(treatment_rate) - float(control_rate))


def serialize_full_evaluation(evaluation: FullEvaluation) -> dict[str, Any]:
    payload = asdict(evaluation)
    payload["evaluator_version"] = EVALUATOR_VERSION
    return payload


def _clean_raw_answer(raw: str) -> str:
    text = str(raw or "").strip()
    text = text.replace("\r\n", "\n")
    return text


def _answer_line(text: str) -> str:
    for raw_line in text.splitlines():
        match = _ANSWER_LINE_RE.match(raw_line)
        if match:
            return match.group(1)
    return ""


def _contains_answer_label(text: str) -> bool:
    return any(re.match(r"^\s*(?:#{1,6}\s*)?(?:final\s+answer|answer|short\s+answer)\s*:\s*$", line, re.IGNORECASE) for line in text.splitlines())


def _first_answer_block(text: str) -> str:
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if cleaned_lines:
                break
            continue
        if stripped.startswith("#"):
            continue
        if stripped.lower() in {"sources", "references"}:
            break
        if stripped.lower().startswith(("source:", "sources:", "references:")):
            break
        cleaned_lines.append(stripped)
        if len(" ".join(cleaned_lines).split()) >= 40:
            break
    return " ".join(cleaned_lines)


def _clean_candidate(candidate: str) -> str:
    candidate = str(candidate or "").strip()
    candidate = _SOURCE_MARK_RE.sub(" ", candidate)
    candidate = re.sub(r"\s+", " ", candidate)
    candidate = candidate.strip(" \t\n\r`*_")
    return candidate.strip()


def _strip_leadin(candidate: str) -> str:
    leadins = [
        r"^(?:the\s+)?(?:final\s+)?answer\s+(?:is|was|would\s+be)\s+",
        r"^it\s+(?:is|was|would\s+be)\s+",
        r"^this\s+(?:is|was)\s+",
    ]
    result = candidate
    for pattern in leadins:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE).strip()
    return result


def _is_refusal(candidate: str) -> bool:
    normalized = normalize_answer(candidate)
    refusal_prefixes = (
        "i cannot answer",
        "i cant answer",
        "cannot answer",
        "can not answer",
        "unable to answer",
        "not enough information",
        "there is not enough information",
        "the answer cannot be determined",
        "no answer can be determined",
        "no specific answer",
        "no specific",
        "unknown",
    )
    return normalized.startswith(refusal_prefixes)


def _answer_is_refusal_or_missing(candidate: str) -> bool:
    normalized = normalize_answer(candidate)
    return _is_refusal(candidate) or normalized in {"none", "n a", "na", "not available", "unknown"}


def _has_unresolved_conflict(candidate: str) -> bool:
    text = " ".join(str(candidate or "").split())
    lower = text.lower()
    if re.search(r"\b(?:either|could be|may be|might be|possibly)\b.{0,100}\bor\b", lower):
        return True
    if re.search(r"\b(?:two|multiple|several)\s+(?:possible|plausible)\s+answers\b", lower):
        return True
    if "conflicting answers" in lower or "cannot distinguish between" in lower:
        return True
    return False


def _question_guided_shortening(question: str, candidate: str) -> tuple[str, str]:
    question_norm = normalize_answer(question)
    one_sentence = _first_sentence(candidate)

    if _asks_yes_no(question_norm):
        yes_no = _extract_yes_no(candidate)
        if yes_no:
            return yes_no, "yes_no_answer"

    date = _extract_date_or_year(question_norm, candidate)
    if date:
        return date, "date_or_year_answer"

    quoted = _quoted_title(candidate, question_norm)
    if quoted:
        return quoted, "quoted_title_answer"

    relation = _relation_answer(candidate, question_norm)
    if relation:
        return relation, "relation_object_answer"

    prefix = _identity_prefix(one_sentence)
    if prefix:
        return prefix, "identity_prefix_answer"

    if _looks_like_list_question(question_norm):
        return _strip_terminal_punctuation(one_sentence), "list_or_sentence_answer"

    if _looks_short(candidate):
        return _strip_terminal_punctuation(candidate), "short_answer_line"

    return _strip_terminal_punctuation(one_sentence), "first_sentence_answer"


def _asks_yes_no(question_norm: str) -> bool:
    return question_norm.split(" ", 1)[0] in {
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "can",
        "could",
    }


def _extract_yes_no(candidate: str) -> str | None:
    norm = normalize_answer(candidate)
    if re.search(r"\byes\b", norm):
        return "yes"
    if re.search(r"\bno\b", norm) or re.search(r"\bnot\b", norm) or re.search(r"\bdifferent\b", norm):
        return "no"
    if "both" in norm and any(token in norm for token in ("same", "from", "are", "originated")):
        return "yes"
    if any(phrase in norm for phrase in ("same country", "as well", "also from", "both from", "both originated")):
        return "yes"
    return None


def _extract_date_or_year(question_norm: str, candidate: str) -> str | None:
    asks_date = any(word in question_norm for word in (" date ", " when ", " day ", " premiered", " released"))
    asks_year = any(word in question_norm.split() for word in ("year", "born", "founded", "established", "published"))
    if asks_date:
        date_match = _MONTH_DATE_RE.search(candidate)
        if date_match:
            return _strip_terminal_punctuation(date_match.group(0))
    if asks_year or asks_date:
        years = _YEAR_RE.findall(candidate)
        if years:
            return years[0]
    return None


def _quoted_title(candidate: str, question_norm: str) -> str | None:
    if not any(word in question_norm for word in ("book", "film", "album", "song", "novel", "magazine", "series")):
        return None
    quotes = re.findall(r"['\"]([^'\"]{2,120})['\"]", candidate)
    return quotes[0].strip() if quotes else None


def _relation_answer(candidate: str, question_norm: str) -> str | None:
    sentence = _first_sentence(candidate)
    patterns: list[str] = []
    if any(word in question_norm for word in ("named after", "created after", "called after")):
        patterns.extend(
            [
                r"\b(?:named|created|called)\s+after\s+(?:the\s+)?(.+?)(?:\s+programming\s+language|\s+language|[,.;]|$)",
                r"\bafter\s+(?:the\s+)?(.+?)(?:\s+programming\s+language|\s+language|[,.;]|$)",
            ]
        )
    if any(word in question_norm for word in ("located", "where", "county", "city", "country", "state")):
        patterns.extend(
            [
                r"\b(?:located|situated|found|buried)\s+(?:in|on|at)\s+(.+?)(?:[.;]|$)",
                r"\b(?:in|on|at|from)\s+([A-Z][^.;]{1,80}?)(?:[.;]|$)",
            ]
        )
    if "from" in question_norm or "nationality" in question_norm:
        patterns.extend([r"\bfrom\s+(?:the\s+)?(.+?)(?:[.;]|$)", r"\b(?:are|is|was|were)\s+(.+?)(?:[.;]|$)"])
    patterns.extend(
        [
            r"\b(?:was|were|is|are)\s+(?:the\s+)?(.+?)(?:[.;]|$)",
            r"\b(?:by|as)\s+([A-Z][^.;]{1,120}?)(?:[.;]|$)",
        ]
    )
    for pattern in patterns:
        for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
            prefix_window = sentence[max(0, match.start() - 16) : match.start()].lower()
            if re.search(r"\b(?:not|never|no)\s+$", prefix_window):
                continue
            answer = _clean_relation_object(match.group(1))
            if _reasonable_relation_object(answer):
                return answer
    return None


def _clean_relation_object(answer: str) -> str:
    answer = _clean_candidate(answer)
    answer = re.sub(r"^(?:an?|the)\s+", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\s+(?:who|which|that)\b.*$", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\s+(?:in|on|at)\s+(?:the\s+)?\d{3,4}\b.*$", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\s+(?:programming\s+language|language)$", "", answer, flags=re.IGNORECASE)
    return _strip_terminal_punctuation(answer)


def _reasonable_relation_object(answer: str) -> bool:
    if not answer:
        return False
    norm = normalize_answer(answer)
    if len(norm.split()) > 16:
        return False
    if norm in {"answer", "company", "person", "place", "yes", "no"}:
        return False
    return True


def _identity_prefix(sentence: str) -> str | None:
    patterns = [
        r"^(.{2,120}?)\s+(?:is|was|are|were)\s+(?:the|an?|both)\b",
        r"^(.{2,120}?)\s*,\s+(?:who|which)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, sentence, flags=re.IGNORECASE)
        if match:
            prefix = _strip_terminal_punctuation(_clean_candidate(match.group(1)))
            if 1 <= len(normalize_answer(prefix).split()) <= 10:
                return prefix
    return None


def _first_sentence(candidate: str) -> str:
    text = _clean_candidate(candidate)
    boundary = re.search(r"(?<!\b[A-Z])(?<!\b[A-Z][a-z])(?<=[.!?])\s+", text)
    if boundary:
        return text[: boundary.start() + 1]
    return text


def _looks_short(candidate: str) -> bool:
    return len(normalize_answer(candidate).split()) <= 12 and "\n" not in str(candidate or "")


def _looks_like_list_question(question_norm: str) -> bool:
    return any(phrase in question_norm for phrase in ("which cities", "what cities", "what are", "list", "major cities"))


def _strip_terminal_punctuation(text: str) -> str:
    return _clean_candidate(text).strip(" .;:")


def _answer_choices(gold: str, aliases: Sequence[str] | None = None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in [gold, *(aliases or [])]:
        text = str(value or "").strip()
        norm = normalize_answer(text)
        if text and norm and norm not in seen:
            seen.add(norm)
            out.append(text)
    return out


def _score_yes_no(pred_norm: str, choices: Sequence[str]) -> tuple[str, str | None] | None:
    gold_norms = [normalize_answer(choice) for choice in choices]
    if not any(gold in {"yes", "no"} for gold in gold_norms):
        return None
    pred_yes = pred_norm == "yes" or pred_norm.startswith("yes ")
    pred_no = pred_norm == "no" or pred_norm.startswith("no ") or re.search(r"\bnot\b", pred_norm)
    for choice, choice_norm in zip(choices, gold_norms):
        if choice_norm == "yes" and pred_yes:
            return "correct", choice
        if choice_norm == "no" and pred_no and not pred_yes:
            return "correct", choice
    return "incorrect", None


def _score_list_answer(prediction: str, choices: Sequence[str]) -> tuple[str, str, str | None] | None:
    for choice in choices:
        gold_items = _split_list_items(choice)
        if len(gold_items) < 2:
            continue
        pred_norm = normalize_answer(prediction)
        matched = [item for item in gold_items if _contains_answer_phrase(pred_norm, normalize_answer(item))]
        if len(matched) == len(gold_items):
            return "correct", "complete_list_answer", choice
        if matched:
            return "incorrect", "partial_list_answer", choice
        return "incorrect", "missing_list_items", choice
    return None


def _split_list_items(text: str) -> list[str]:
    clean = str(text or "")
    clean = re.sub(r"\band\b", ",", clean, flags=re.IGNORECASE)
    parts = [part.strip(" .;") for part in re.split(r"[,;/]", clean)]
    return [part for part in parts if normalize_answer(part)]


_NATIONALITY_EQUIVALENTS = {
    "american": {"united states", "usa", "us", "u s", "united states of america"},
    "british": {"united kingdom", "uk", "england", "english", "scotland", "wales"},
    "english": {"england", "british", "united kingdom", "uk"},
    "canadian": {"canada"},
    "australian": {"australia"},
    "french": {"france"},
    "german": {"germany"},
    "spanish": {"spain"},
    "italian": {"italy"},
}


def _semantic_equivalent(choice_norm: str, pred_norm: str) -> bool:
    if _entity_suffix_equivalent(choice_norm, pred_norm):
        return True
    equivalents = _NATIONALITY_EQUIVALENTS.get(choice_norm)
    if equivalents and any(_contains_answer_phrase(pred_norm, equivalent) for equivalent in equivalents):
        return True
    for nationality, countries in _NATIONALITY_EQUIVALENTS.items():
        if pred_norm == nationality and choice_norm in countries:
            return True
    return False


_ENTITY_SUFFIXES = (
    "football club",
    "fc",
    "f c",
    "club",
    "incorporated",
    "inc",
    "company",
    "companies",
    "corporation",
    "corp",
    "limited",
    "ltd",
    "plc",
)


def _entity_suffix_equivalent(choice_norm: str, pred_norm: str) -> bool:
    choice_base = _strip_entity_suffix(choice_norm)
    pred_base = _strip_entity_suffix(pred_norm)
    if choice_base and pred_base and choice_base == pred_base:
        return True
    return False


def _strip_entity_suffix(text: str) -> str:
    result = text
    changed = True
    while changed:
        changed = False
        for suffix in _ENTITY_SUFFIXES:
            if result.endswith(" " + suffix):
                result = result[: -len(suffix) - 1].strip()
                changed = True
    return result


def _contains_answer_phrase(haystack_norm: str, needle_norm: str) -> bool:
    if not haystack_norm or not needle_norm:
        return False
    return re.search(rf"(?<!\w){re.escape(needle_norm)}(?!\w)", haystack_norm) is not None


def _negated_or_alternative_mention(pred_norm: str, choice_norm: str) -> bool:
    if not _contains_answer_phrase(pred_norm, choice_norm):
        return False
    token_pattern = re.escape(choice_norm)
    neg_before = rf"\b(?:not|no|never|without|incorrectly|rather than|instead of)\b(?:\W+\w+){{0,6}}\W+{token_pattern}\b"
    neg_after = rf"\b{token_pattern}\b(?:\W+\w+){{0,6}}\W+\b(?:not|incorrect|false|rather than|instead of)\b"
    rejected = rf"\bnot\b(?:\W+\w+){{0,10}}\b{token_pattern}\b(?:\W+\w+){{0,10}}\bbut\b"
    return bool(
        re.search(neg_before, pred_norm)
        or re.search(neg_after, pred_norm)
        or re.search(rejected, pred_norm)
    )


def _multiple_candidate_answer_contains_gold(pred_norm: str, choice_norm: str) -> bool:
    if not _contains_answer_phrase(pred_norm, choice_norm):
        return False
    if " or " in f" {pred_norm} " and not pred_norm.startswith(("yes ", "no ")):
        return True
    multi_markers = (
        " include ",
        " includes ",
        " included ",
        " including ",
        " among ",
        " one of ",
        " several ",
        " multiple ",
    )
    has_marker = any(marker in f" {pred_norm} " for marker in multi_markers)
    if not has_marker:
        return False
    after = pred_norm.split(choice_norm, 1)[1] if choice_norm in pred_norm else ""
    before = pred_norm.split(choice_norm, 1)[0] if choice_norm in pred_norm else ""
    return any(token in after for token in (" and ", " among", " others", " as well")) or before.count(" ") > 4


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()
