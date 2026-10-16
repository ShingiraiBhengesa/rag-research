"""Prespecified semantic check s |= f for step 2.

Uses the locally cached NLI cross-encoder pinned in config (deterministic, CPU
is sufficient). A fact is span-backed iff `entailment` is the argmax label
and its probability is >= NLI_ENTAIL_MIN_PROB. Span presence alone never
suffices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from prematurestop.pipelines.bridge_mechanism import config as C


@dataclass(frozen=True)
class EntailVerdict:
    entails: bool
    entail_prob: float
    label: str


class Entailer(Protocol):
    def entails(self, *, premise: str, hypothesis: str) -> EntailVerdict: ...


class NliEntailer:
    def __init__(self, model_id: str = C.NLI_MODEL_ID, revision: str = C.NLI_REVISION, device: str = "cpu"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision).to(device).eval()
        self.device = device
        self.id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        if "entailment" not in self.id2label.values():
            raise RuntimeError(f"NLI model has no entailment label: {self.id2label}")

    def entails(self, *, premise: str, hypothesis: str) -> EntailVerdict:
        enc = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with self._torch.no_grad():
            probs = self._torch.softmax(self.model(**enc).logits[0], dim=-1).tolist()
        top = max(range(len(probs)), key=probs.__getitem__)
        ent_idx = next(i for i, lab in self.id2label.items() if lab == "entailment")
        p = float(probs[ent_idx])
        return EntailVerdict(entails=(top == ent_idx and p >= C.NLI_ENTAIL_MIN_PROB), entail_prob=round(p, 6),
                             label=self.id2label[top])
