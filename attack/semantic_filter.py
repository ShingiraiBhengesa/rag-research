"""Strict semantic-equivalence filters."""
import re
from typing import Any, Dict, List, Set
import torch
import torch.nn.functional as F


# ----- Filter 4: entity preservation -----
class EntityPreservationFilter:
    ENTITY_TYPES = {"PERSON", "ORG", "GPE", "LOC", "FAC", "EVENT",
                    "WORK_OF_ART", "PRODUCT", "NORP"}

    def __init__(self, nlp):
        self.nlp = nlp

    @staticmethod
    def _normalize(text):
        return re.sub(r"\s+", " ",
                      re.sub(r"[^\w\s]", "", text.lower())).strip()

    def _extract_entities(self, text):
        doc = self.nlp(text)
        ents = set()
        for ent in doc.ents:
            if ent.label_ in self.ENTITY_TYPES:
                normalized = self._normalize(ent.text)
                if normalized:
                    ents.add(normalized)
        return ents

    @staticmethod
    def _entity_present(target, entity_set):
        for e in entity_set:
            if target == e or target in e or e in target:
                return True
        return False

    def evaluate(self, original, candidate):
        orig_ents = self._extract_entities(original)
        cand_ents = self._extract_entities(candidate)
        new_in_cand = [e for e in cand_ents
                       if not self._entity_present(e, orig_ents)]
        missing = [e for e in orig_ents
                   if not self._entity_present(e, cand_ents)]
        passed = len(new_in_cand) == 0 and len(missing) == 0
        reason = ""
        if new_in_cand: reason = f"hallucinated_entities: {new_in_cand}"
        elif missing: reason = f"missing_entities: {missing}"
        return {
            "passed": passed, "reason": reason,
            "original_entities": list(orig_ents),
            "candidate_entities": list(cand_ents),
            "hallucinated_entities": new_in_cand,
            "missing_entities": missing,
        }


# ----- Filter 5: bidirectional NLI -----
class NLIBidirectionalFilter:
    DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-base"

    def __init__(self, model_name=None, device="cpu", entailment_threshold=0.7):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model_name = model_name or self.DEFAULT_MODEL
        print(f"[nli] loading {model_name} on {device}, threshold={entailment_threshold}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device).eval()
        self.device = device
        self.threshold = entailment_threshold
        self.entailment_idx = 1

    @torch.no_grad()
    def _entailment_prob(self, premise, hypothesis):
        inp = self.tokenizer(premise, hypothesis, return_tensors="pt",
                             truncation=True, max_length=256, padding=True)
        inp = {k: v.to(self.device) for k, v in inp.items()}
        logits = self.model(**inp).logits[0]
        probs = F.softmax(logits, dim=-1)
        return float(probs[self.entailment_idx])

    def evaluate(self, original, candidate):
        forward = self._entailment_prob(original, candidate)
        reverse = self._entailment_prob(candidate, original)
        passed = forward >= self.threshold and reverse >= self.threshold
        reason = ""
        if forward < self.threshold and reverse < self.threshold:
            reason = "both_directions_fail"
        elif forward < self.threshold:
            reason = "forward_fails_likely_hallucination"
        elif reverse < self.threshold:
            reason = "reverse_fails_likely_dropped_constraint"
        return {
            "passed": passed, "reason": reason,
            "forward_entailment": forward,
            "reverse_entailment": reverse,
            "threshold": self.threshold,
        }


# ----- Filter 6: question-form preservation -----
class QuestionFormFilter:
    """Reject candidates that aren't questions (uncertainty injection, declaratives).

    The original is always interrogative. The rephrasing must remain so:
      - Must end with '?' (after stripping whitespace/closing punct)
      - Must NOT contain assertion phrases like 'it is not known', 'remains unknown'
      - Must NOT start with declarative-only stems like 'It is', 'No one', 'Nobody'
    """

    ASSERTION_PHRASES = [
        "it is not known", "it is not clear", "it is unclear",
        "is not known", "is not clear", "remains unknown",
        "remains a mystery", "no one knows", "nobody knows",
        "we do not know", "we don't know", "one might wonder",
        "is unknown", "are unknown",
    ]

    DECLARATIVE_STEMS = ["it is ", "there is ", "no one ", "nobody ",
                         "one might ", "one could ", "one wonders"]

    def evaluate(self, original, candidate):
        cand_lower = candidate.strip().lower()
        # Must end with question mark
        ends_question = cand_lower.rstrip(".!\"' ").endswith("?")
        # No assertion phrases
        has_assertion = any(p in cand_lower for p in self.ASSERTION_PHRASES)
        # Doesn't start with declarative-only stem (only flag if no '?' anywhere
        # in the rephrase — some embed-style rephrases start "What" but
        # the test is harmless because they always contain '?')
        starts_declarative = any(cand_lower.startswith(s)
                                 for s in self.DECLARATIVE_STEMS)
        passed = ends_question and not has_assertion and not starts_declarative
        reasons = []
        if not ends_question: reasons.append("not_a_question")
        if has_assertion: reasons.append("assertion_phrase")
        if starts_declarative: reasons.append("declarative_stem")
        return {"passed": passed, "reason": "|".join(reasons),
                "ends_with_question_mark": ends_question,
                "has_assertion_phrase": has_assertion,
                "starts_with_declarative_stem": starts_declarative}


# ----- Filter 7: qualifier blocklist -----
class QualifierBlocklistFilter:
    """Reject candidates that introduce 'puffy' qualifier adjectives not in original.

    LLMs love to add words like 'renowned', 'iconic', 'historic'. These don't
    introduce new entities (so entity filter passes) but they do add unverifiable
    properties to entities. We block any of a small list of known offenders.
    """

    BLOCKLIST = {
        "renowned", "famous", "famed", "notable", "noted", "notably",
        "iconic", "influential", "ubiquitous", "historic", "historical",
        "celebrated", "prestigious", "legendary", "acclaimed", "esteemed",
        "distinguished", "remarkable", "remarkably", "significant",
        "significantly", "specific", "specifically", "particularly",
        "originally", "subsequently", "ultimately",
    }

    def __init__(self, nlp=None):
        # nlp not strictly needed; tokens via simple lowercasing/splitting OK
        self.nlp = nlp

    @staticmethod
    def _tokenize(text):
        return set(re.findall(r"\b[a-z]+\b", text.lower()))

    def evaluate(self, original, candidate):
        orig_tokens = self._tokenize(original)
        cand_tokens = self._tokenize(candidate)
        introduced_qualifiers = sorted([
            w for w in cand_tokens & self.BLOCKLIST
            if w not in orig_tokens
        ])
        passed = len(introduced_qualifiers) == 0
        reason = ""
        if introduced_qualifiers:
            reason = f"introduced_qualifiers: {introduced_qualifiers}"
        return {"passed": passed, "reason": reason,
                "introduced_qualifiers": introduced_qualifiers}


# ----- Filter 8: categorical descriptor preservation -----
class CategoricalDescriptorFilter:
    """Reject candidates that add categorical descriptors not in original.

    Catches patterns like:
      - "X, which is a type of portable music player"  (when 'portable music player' not in orig)
      - "Walt Disney, who was an American animator"     (when 'American animator' not in orig)
      - "the river, which is a body of water"           (when 'body of water' not in orig)

    Allows:
      - "Mara Region, which is a region"  (because 'region' is in original)
      - "X, who is the person that did Y"  (no category, just reference back)
    """
    import re as _re
    PATTERN = _re.compile(
        r"\b(which|who|that)\s+(is|was|were)\s+(a|an)\s+([^,.?]+)",
        _re.IGNORECASE
    )

    def __init__(self, nlp):
        self.nlp = nlp

    def _content_lemmas(self, text):
        doc = self.nlp(text)
        return {t.lemma_.lower() for t in doc
                if t.pos_ in ("NOUN", "PROPN", "ADJ", "VERB")
                and not t.is_stop and t.text.strip()}

    def evaluate(self, original, candidate):
        orig_lemmas = self._content_lemmas(original)
        matches = self.PATTERN.findall(candidate)
        introduced = []
        for m in matches:
            descriptor = m[3].strip()
            descriptor_lemmas = self._content_lemmas(descriptor)
            new_in_descriptor = descriptor_lemmas - orig_lemmas
            if new_in_descriptor:
                introduced.append({"descriptor": descriptor,
                                   "new_content": list(new_in_descriptor)})
        passed = len(introduced) == 0
        reason = ""
        if introduced:
            reason = f"categorical_descriptor_with_new_content: {introduced}"
        return {"passed": passed, "reason": reason,
                "introduced_descriptors": introduced}

