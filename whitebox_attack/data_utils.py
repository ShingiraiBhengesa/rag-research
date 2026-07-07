"""
data_utils.py — Dataset loading and perturbation validation for Attack A.

Role in the pipeline
--------------------
Pure I/O layer: loads multi-hop QA examples from four benchmark datasets and
exposes the two validation helpers (is_ascii, phrase_valid) used by every
attack script to screen generated perturbation text before it is appended to
a retrieved document. No model, torch, or GPU dependency — importable offline.

Datasets
--------
HotpotQA       "distractor" split, validation set (Carter et al., 2018).
               Bridge structure: supporting_facts gives two titles in order;
               first title = first-hop doc, second title = missing-hop doc.
2WikiMultiHopQA  validation split (Ho et al., 2020).
               Same two-title bridge structure via supporting_facts.
MuSiQue        "answerable" split, validation set (Trivedi et al., 2022).
               Explicit question_decomposition gives paragraph_support_idx
               for each hop, eliminating the title-matching heuristic.
MultiHopRAG    training split (Tang et al., 2024).
               evidence_list[0] = first-hop doc, evidence_list[1] = missing
               hop; no explicit bridge decomposition so we use index order.

Validation filters (applied at load time AND by attack scripts)
---------------------------------------------------------------
ASCII filter  : reject any example or perturbation containing non-ASCII
               characters to prevent tokenizer fragmentation artefacts that
               could make individual cue tokens unrepresentative.
UNIVERSAL_PATTERNS : banned phrases that would trigger tool-call injection or
               agent-override effects irrelevant to the gap-assessment attack
               surface (e.g. "ignore instruction", "call resultcomplete").
FACTONLY_PATTERNS  : banned closure/completion words ("settle", "complete",
               "sufficient", …) that would confound the cue effect with
               explicit closure assertions. The attack cues must work through
               factual-coverage framing only, not by literally telling the
               agent it has enough information.

N_SAMPLE per dataset: configured externally via N_SAMPLE env var (default 20
for smoke, 450 for Stage 1 full runs). load_retrieval_control_pool returns
n_sample examples per source, so the combined pool is n_sample × n_sources.
"""
import re
from datasets import load_dataset


# ── Validation constants ──────────────────────────────────────────────────────

# These phrases would make a perturbation an injection attack (bypassing the
# agent's reasoning) rather than a cue attack (biasing its inference). Kept
# separate from FACTONLY so scripts can apply them independently.
BANNED_WORDS_UNIVERSAL = [
    "call resultcomplete", "do not search", "stop searching", "ignore instruction",
    "use this answer", "tool", "assistant", "agent",
]

# Explicit closure/completion vocabulary. If the cue contains these, success
# is trivially caused by the direct command rather than the epistemic framing.
BANNED_WORDS_FACTONLY = [
    "settle", "establish", "answer", "complete", "sufficient", "no further", "research question",
]

UNIVERSAL_PATTERNS = [re.compile(r"\b" + re.escape(b) + r"\b") for b in BANNED_WORDS_UNIVERSAL]
FACTONLY_PATTERNS  = [re.compile(r"\b" + re.escape(b) + r"\b") for b in BANNED_WORDS_FACTONLY]

# Unicode normalisation map: convert common "smart" punctuation to ASCII
# equivalents so the ASCII filter does not reject otherwise-valid examples
# that HuggingFace datasets return with typographic quotes or en-dashes.
UNICODE_TO_ASCII = {
    "–": "-", "—": "--", "'": "'", "'": "'",
    """: '"', """: '"', " ": " ", "…": "...",
}


# ── Validation helpers ────────────────────────────────────────────────────────

def normalize_text(s):
    """Replace common Unicode punctuation with ASCII equivalents before the
    is_ascii check, so we don't discard examples on typographic quotes alone."""
    for u, a in UNICODE_TO_ASCII.items():
        s = s.replace(u, a)
    return s


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def phrase_valid(phrase, patterns):
    """Return True iff phrase passes the ASCII check and matches none of the
    compiled banned-word patterns (case-insensitive, whole-word)."""
    if not is_ascii(phrase):
        return False
    low = phrase.lower()
    return not any(p.search(low) for p in patterns)


# ── Perturbation building helper ──────────────────────────────────────────────

def first_sentence(para):
    """Extract the first sentence from a paragraph using the '. [A-Z]' boundary
    heuristic (same heuristic as the original act_poison_real_data.py so results
    are reproducible). Used by build_cues() in both attack scripts to produce
    the natural-language prefix that precedes the rhetorical cue suffix.

    The attack design requires appending the *real* gold second-hop fact so that
    any success is attributable to the cue suffix, not to injected false content.
    first_sentence(second_para) gives the most salient piece of that fact without
    over-loading the context with the full paragraph."""
    m = re.search(r"\.\s+[A-Z]", para)
    end = (m.start() + 1) if m else len(para.rstrip("."))
    sent = para[:end].strip()
    if not sent.endswith("."):
        sent += "."
    return sent


# ── Dataset loaders ───────────────────────────────────────────────────────────
# Each loader returns a list of dicts with these keys:
#   name              : unique example identifier from the source dataset
#   topic             : the multi-hop question (agent's search topic)
#   body              : first-hop supporting paragraph (the observation injected
#                       into the simulated retrieval result; the agent has this)
#   second_para       : second-hop supporting paragraph (missing-hop fact; the
#                       agent does NOT have this — it is what the attack injects)
#   first_title       : entity name for the first-hop document
#   second_title      : entity name for the missing-hop document
#   answer            : gold answer string (not used by Stage 1 attacks)
#   source            : dataset key string ("hotpot", "2wiki", "musique", "multihoprag")
#   gold_missing_hop_terms  : list of strings identifying the missing hop
#   distractor_terms        : list of strings for the first-hop entity (distractors
#                             for Attack B; not used by Attack A)

def _load_hotpot(n_sample):
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    examples = []
    for ex in ds:
        titles = ex["supporting_facts"]["title"]
        uniq = list(dict.fromkeys(titles))
        if len(uniq) != 2:
            continue
        t1_raw, t2_raw = uniq
        ctx_titles = ex["context"]["title"]
        ctx_sents  = ex["context"]["sentences"]
        if t1_raw not in ctx_titles or t2_raw not in ctx_titles:
            continue
        i1, i2 = ctx_titles.index(t1_raw), ctx_titles.index(t2_raw)
        t1 = normalize_text(t1_raw)
        t2 = normalize_text(t2_raw)
        p1 = normalize_text(re.sub(r"\s+", " ", " ".join(ctx_sents[i1]).strip()))
        p2 = normalize_text(re.sub(r"\s+", " ", " ".join(ctx_sents[i2]).strip()))
        if not (40 <= len(p1) <= 800 and 20 <= len(p2) <= 800):
            continue
        if not is_ascii(p1) or not phrase_valid(p1, UNIVERSAL_PATTERNS):
            continue
        examples.append(dict(
            name=ex["id"], topic=normalize_text(ex["question"]),
            body=p1, second_para=p2, first_title=t1, second_title=t2,
            answer=ex["answer"], source="hotpot",
            gold_missing_hop_terms=[t2], distractor_terms=[t1],
        ))
        if len(examples) >= n_sample:
            break
    return examples


def _load_2wiki(n_sample):
    ds = load_dataset("voidful/2WikiMultihopQA", split="validation")
    examples = []
    for ex in ds:
        titles = [t for t, _ in ex["supporting_facts"]]
        uniq = list(dict.fromkeys(titles))
        if len(uniq) != 2:
            continue
        t1_raw, t2_raw = uniq
        ctx_map = {t: s for t, s in ex["context"]}
        if t1_raw not in ctx_map or t2_raw not in ctx_map:
            continue
        t1 = normalize_text(t1_raw)
        t2 = normalize_text(t2_raw)
        p1 = normalize_text(re.sub(r"\s+", " ", " ".join(ctx_map[t1_raw]).strip()))
        p2 = normalize_text(re.sub(r"\s+", " ", " ".join(ctx_map[t2_raw]).strip()))
        if not (40 <= len(p1) <= 800 and 20 <= len(p2) <= 800):
            continue
        if not is_ascii(p1) or not phrase_valid(p1, UNIVERSAL_PATTERNS):
            continue
        examples.append(dict(
            name=ex["_id"], topic=normalize_text(ex["question"]),
            body=p1, second_para=p2, first_title=t1, second_title=t2,
            answer=ex["answer"], source="2wiki",
            gold_missing_hop_terms=[t2], distractor_terms=[t1],
        ))
        if len(examples) >= n_sample:
            break
    return examples


def _load_musique(n_sample):
    ds = load_dataset("bdsaglam/musique", "answerable", split="validation")
    examples = []
    for ex in ds:
        decomp = ex["question_decomposition"]
        if len(decomp) != 2:
            continue
        paras = ex["paragraphs"]
        i1 = decomp[0]["paragraph_support_idx"]
        i2 = decomp[1]["paragraph_support_idx"]
        if i1 is None or i2 is None or i1 >= len(paras) or i2 >= len(paras):
            continue
        p1d, p2d = paras[i1], paras[i2]
        if not p1d["is_supporting"] or not p2d["is_supporting"]:
            continue
        t1 = normalize_text(p1d["title"])
        t2 = normalize_text(p2d["title"])
        p1 = normalize_text(re.sub(r"\s+", " ", p1d["paragraph_text"].strip()))
        p2 = normalize_text(re.sub(r"\s+", " ", p2d["paragraph_text"].strip()))
        if not (40 <= len(p1) <= 800 and 20 <= len(p2) <= 800):
            continue
        if not is_ascii(p1) or not phrase_valid(p1, UNIVERSAL_PATTERNS):
            continue
        bridge = normalize_text(decomp[0]["answer"])
        non_sup = [normalize_text(p["title"]) for p in paras if not p["is_supporting"]]
        examples.append(dict(
            name=ex["id"], topic=normalize_text(ex["question"]),
            body=p1, second_para=p2, first_title=t1, second_title=t2,
            answer=ex["answer"], source="musique",
            gold_missing_hop_terms=[bridge, t2],
            distractor_terms=[non_sup[0]] if non_sup else [t1],
        ))
        if len(examples) >= n_sample:
            break
    return examples


def _load_multihoprag(n_sample):
    # MultiHopRAG has no explicit hop decomposition; evidence_list order is
    # treated as first-hop / second-hop, which matches how the dataset is
    # constructed (evidence_list[0] is the most directly retrieved fact).
    ds = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train")
    examples = []
    for i, ex in enumerate(ds):
        ev = ex["evidence_list"]
        if len(ev) < 2:
            continue
        if len({e["title"] for e in ev[:2]}) != 2:
            continue
        t1 = normalize_text(ev[0]["title"])
        t2 = normalize_text(ev[1]["title"])
        p1 = normalize_text(re.sub(r"\s+", " ", ev[0]["fact"].strip()))
        p2 = normalize_text(re.sub(r"\s+", " ", ev[1]["fact"].strip()))
        if not (40 <= len(p1) <= 800 and 20 <= len(p2) <= 800):
            continue
        if not is_ascii(p1) or not phrase_valid(p1, UNIVERSAL_PATTERNS):
            continue
        examples.append(dict(
            name=f"mhrag_{i}", topic=normalize_text(ex["query"]),
            body=p1, second_para=p2, first_title=t1, second_title=t2,
            answer=ex["answer"], source="multihoprag",
            gold_missing_hop_terms=[t2], distractor_terms=[t1],
        ))
        if len(examples) >= n_sample:
            break
    return examples


_LOADERS = {
    "hotpot":      _load_hotpot,
    "2wiki":       _load_2wiki,
    "musique":     _load_musique,
    "multihoprag": _load_multihoprag,
}


def load_retrieval_control_pool(n_sample=60, dataset_src="all"):
    """Load n_sample examples per dataset and return them as a flat list.

    dataset_src: "all" to load all four datasets, or a comma-separated subset
    of {"hotpot", "2wiki", "musique", "multihoprag"} for targeted runs.

    Each example represents a case where the agent has been given the first-hop
    supporting document as its search result (body) and must decide whether to
    search further or call ResearchComplete. The gold second-hop paragraph
    (second_para) is withheld from the agent; it is what the attack scripts
    inject as an appended cue to test whether the agent can be fooled into
    treating the first-hop result as sufficient.
    """
    sources = list(_LOADERS) if dataset_src == "all" else dataset_src.split(",")
    examples = []
    for src in sources:
        examples += _LOADERS[src](n_sample)
    return examples
