"""
metrics.py — Statistical inference helpers for Attack A result analysis.

All functions are pure Python / NumPy with no torch or scipy dependency so they
can be imported in any environment (offline analysis, CI, table-building scripts)
without triggering a GPU model load.

When to use each test
---------------------
wilson_ci             : primary CI for all reported rates (WGAR, PCR@k, Net@k).
                        Wilson score interval is preferred over normal approximation
                        for proportions; accurate even at low n or extreme rates.
                        Use z=1.96 for 95% CI (default).

mcnemar_test          : compare two conditions on the *same* set of paired examples
                        (e.g. epistemic_confirmation vs search_match_cue on the
                        same eligible subset). Accounts for correlated outcomes.
                        Continuity-corrected; use n_discordant as a sanity check
                        (if very small, the test has low power).

paired_bootstrap_ci   : non-parametric alternative to McNemar when the outcome is
                        continuous (e.g. delta_J_reflect) or when the sample size
                        is too small for chi-squared approximation. Also handles
                        arbitrary stat_fn (e.g. median instead of mean).

em_f1                 : SQuAD-style answer quality metrics for Stage 2 evaluation
                        (real search backend). Not used in Stage 1 logged outputs.

branch_score_predictiveness : validates that the white-box branch margin (delta_J_reflect)
                        actually predicts greedy-generation success. If AUC is near
                        0.5 or Spearman near 0, the branch score is not a reliable
                        proxy and WGAR/PCR numbers should be reported without it.
"""
import math
import re
import string
import numpy as np


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
    a = np.asarray(a_successes, dtype=bool)
    b = np.asarray(b_successes, dtype=bool)
    if len(a) != len(b):
        raise ValueError("a_successes and b_successes must be the same length (paired)")
    n01 = int(np.sum((~a) & b))   # A fails, B succeeds
    n10 = int(np.sum(a & (~b)))   # A succeeds, B fails
    n_discordant = n01 + n10
    if n_discordant == 0:
        return (0.0, 1.0, 0)
    stat = (abs(n01 - n10) - 1) ** 2 / n_discordant
    p_value = math.erfc(math.sqrt(stat / 2))  # chi2(df=1) survival function
    return (stat, p_value, n_discordant)


def paired_bootstrap_ci(a, b, stat_fn=lambda x: np.mean(x), n_boot=10000, seed=0, alpha=0.05):
    """Paired bootstrap CI for stat_fn(a) - stat_fn(b). a, b are same-length arrays
    of per-example values (e.g. booleans or continuous metrics). Returns
    (point_estimate, lo, hi, p_value_two_sided_vs_zero)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("a and b must be the same length (paired)")
    n = len(a)
    rng = np.random.default_rng(seed)
    point = stat_fn(a) - stat_fn(b)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = stat_fn(a[idx]) - stat_fn(b[idx])
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    p_value = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
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


def _auc(scores, labels):
    """AUC via the rank-sum (Mann-Whitney U) formula -- no sklearn dependency."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg])))
    pos_ranks = ranks[: len(pos)]
    auc = (pos_ranks.sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def _spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def branch_score_predictiveness(margins, labels):
    """Section 7.15: does the white-box branch margin (J / delta-J) predict the
    real greedy-generation success label? margins: array of branch-margin values
    (e.g. delta_J, more-negative = more attack-favoring). labels: array of bool,
    True = attack actually succeeded under greedy generation. Returns a dict with
    auc, spearman, and brier (brier computed on sigmoid(-margin) as the predicted
    attack-success probability, since lower/more-negative margin should predict
    higher attack success)."""
    margins = np.asarray(margins, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    auc = _auc(-margins, labels)  # more-negative margin -> higher predicted success
    spearman = _spearman(margins, labels.astype(float))
    probs = _sigmoid(-margins)
    brier = float(np.mean((probs - labels.astype(float)) ** 2))
    return dict(auc=auc, spearman=spearman, brier=brier, n=len(margins))
