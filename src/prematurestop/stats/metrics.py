"""
metrics.py — Statistical inference helpers for the attack's result analysis.

All functions are pure Python / NumPy with no torch or scipy dependency so they
can be imported in any environment (offline analysis, CI, table-building scripts)
without triggering a GPU model load.

When to use each function
--------------------------
wilson_ci             : primary CI for all reported rates (e.g. PCR@k). Wilson
                        score interval is preferred over normal approximation
                        for proportions; accurate even at low n or extreme
                        rates. Use z=1.96 for 95% CI (default).

mcnemar_test          : compare two conditions on the *same* set of paired
                        examples (e.g. epistemic_confirmation vs
                        search_match_cue on the same eligible subset).
                        Accounts for correlated outcomes. Continuity-corrected;
                        use n_discordant as a sanity check (if very small, the
                        test has low power).

paired_bootstrap_ci   : non-parametric alternative to McNemar when the outcome
                        is continuous or when the sample size is too small for
                        chi-squared approximation. Also handles arbitrary
                        stat_fn (e.g. median instead of mean).

em_f1                 : SQuAD-style exact-match / token-F1 answer quality
                        metrics, used by src/prematurestop/evaluation/answer_scoring.py
                        and src/prematurestop/pipelines/fixed_cue_matrix/production_runner.py.
"""
import math
import re
import string
try:
    import numpy as np
except ModuleNotFoundError:
    np = None


def _require_numpy():
    if np is None:
        raise ModuleNotFoundError("numpy is required for this statistical helper")
    return np


def wilson_ci(successes, n, z=1.96):
    """Wilson score interval for a binomial proportion. Returns (point, lo, hi)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = (center - margin) / denom
    hi = (center + margin) / denom
    return (p, max(0.0, lo), min(1.0, hi))


def mcnemar_test(a_successes, b_successes):
    """Paired McNemar's test (with continuity correction) on two same-length boolean
    sequences indicating per-example success of method A vs method B. Returns
    (statistic, p_value, n_discordant)."""
    np_mod = _require_numpy()
    a = np_mod.asarray(a_successes, dtype=bool)
    b = np_mod.asarray(b_successes, dtype=bool)
    if len(a) != len(b):
        raise ValueError("a_successes and b_successes must be the same length (paired)")
    n01 = int(np_mod.sum((~a) & b))   # A fails, B succeeds
    n10 = int(np_mod.sum(a & (~b)))   # A succeeds, B fails
    n_discordant = n01 + n10
    if n_discordant == 0:
        return (0.0, 1.0, 0)
    stat = (abs(n01 - n10) - 1) ** 2 / n_discordant
    p_value = math.erfc(math.sqrt(stat / 2))  # chi2(df=1) survival function
    return (stat, p_value, n_discordant)


def paired_bootstrap_ci(a, b, stat_fn=None, n_boot=10000, seed=0, alpha=0.05):
    """Paired bootstrap CI for stat_fn(a) - stat_fn(b). a, b are same-length arrays
    of per-example values (e.g. booleans or continuous metrics). Returns
    (point_estimate, lo, hi, p_value_two_sided_vs_zero)."""
    np_mod = _require_numpy()
    if stat_fn is None:
        stat_fn = np_mod.mean
    a = np_mod.asarray(a, dtype=float)
    b = np_mod.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("a and b must be the same length (paired)")
    n = len(a)
    rng = np_mod.random.default_rng(seed)
    point = stat_fn(a) - stat_fn(b)
    diffs = np_mod.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = stat_fn(a[idx]) - stat_fn(b[idx])
    lo, hi = np_mod.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    p_value = 2 * min(np_mod.mean(diffs <= 0), np_mod.mean(diffs >= 0))
    return (point, float(lo), float(hi), float(min(p_value, 1.0)))


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize_answer(s):
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES_RE.sub(" ", s)
    return " ".join(s.split())


def em_f1(pred, gold):
    """Standard SQuAD-style exact-match / token-F1 between a predicted and gold
    answer string. Returns (em, f1), both in {0.0, 1.0} for em and [0, 1] for f1."""
    p_norm = _normalize_answer(pred or "")
    g_norm = _normalize_answer(gold or "")
    em = 1.0 if p_norm == g_norm else 0.0
    p_toks = p_norm.split()
    g_toks = g_norm.split()
    if not p_toks or not g_toks:
        f1 = 1.0 if p_toks == g_toks else 0.0
        return (em, f1)
    common = {}
    for t in p_toks:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    g_counts = {}
    for t in g_toks:
        g_counts[t] = g_counts.get(t, 0) + 1
    for t, c in g_counts.items():
        num_same += min(c, common.get(t, 0))
    if num_same == 0:
        return (em, 0.0)
    precision = num_same / len(p_toks)
    recall = num_same / len(g_toks)
    f1 = 2 * precision * recall / (precision + recall)
    return (em, f1)


