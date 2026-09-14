# Upstream Open Deep Research commit recovery (Phase 1)

**Final label: `EXACT COMMIT VERIFIED`**

**Commit:** [`1f24f1142db24f81e29eb88751985a00ec8ed580`](https://github.com/langchain-ai/open_deep_research/commit/1f24f1142db24f81e29eb88751985a00ec8ed580)
— "Bump the uv group across 1 directory with 2 updates (#276)" — authored
2026-06-19T09:07:36Z.

This supersedes the prior audits' "UNKNOWN" determination for this question.

## Method (network access was available and used, read-only)

1. **Ruled out git-based recovery within this repo**: `git remote -v` shows
   only `origin` (the user's own fork); no remote points at
   `langchain-ai/open_deep_research`. `git reflog --all` and
   `git log --all --source --remotes -- open_deep_research/` show no
   separate clone/fetch trace. Exactly 3 commits ever touch
   `open_deep_research/` in this repo's history, all on `main`, the first
   (and only addition) being `043ecacf` (2026-07-07) — confirming the
   earlier finding that vendoring and the initial attack-pipeline commit
   were bundled together, with no separate "vendor import" commit.
2. **`uv.lock`** confirms `open-deep-research` is recorded as
   `source = { editable = "." }` (a local editable install, not a
   git-sourced dependency) — no VCS revision is recorded there, as expected
   for a vendored copy.
3. **Installed package metadata** (`~/.conda/envs/odr/lib/python3.11/site-packages/open_deep_research-0.0.16.dist-info/direct_url.json`)
   confirms the same: `{"dir_info": {"editable": true}, "url":
   "file:///home/sbhengesa/rag-research/open_deep_research"}` — no
   `vcs_info` block, consistent with a local editable install rather than a
   `pip install git+https://...@<commit>` install (which would have
   recorded the exact commit directly — it did not, which is why this
   required content-based recovery instead).
4. **Shell history**: one unrelated `tar` line referencing an archived
   `.sif` container image; no `git clone`/`pip install` line for this
   package.
5. **Network-based content recovery** (this is what actually resolved the
   question): the upstream repository has no tags/releases. Local
   `pyproject.toml` was first confirmed byte-identical to the LAST upstream
   commit that ever touched that file
   (`d402cc934b350f4a629626e69cde57dd766762c9`, 2026-02-08) — this only
   lower-bounds the vendor date. To pinpoint the exact commit, ~15
   independently-varying dependency version pins recorded in the local
   `uv.lock` (`aiohttp`, `cryptography`, `langsmith`, `starlette`,
   `pyarrow`, `mcp`, `langgraph-checkpoint`, `pyasn1`, `soupsieve`, and
   others) were cross-referenced against the exact sequence of upstream
   dependabot version-bump commits between 2026-02-08 and 2026-08-10 (the
   latest upstream commit at the time of this audit). All ~15 pins converge
   on a single narrow window: strictly after commit `1f24f114...` and
   strictly before the next `uv.lock`-touching commit (`e59ee306`,
   2026-06-26).
6. **Direct verification**: the full `uv.lock` (5,281 lines) fetched from
   commit `1f24f114` was diffed against the local file — **byte-for-byte
   identical**. `pyproject.toml`, `langgraph.json`, `src/security/auth.py`,
   `src/legacy/multi_agent.py`, `src/legacy/graph.py`,
   `src/legacy/legacy.md`, and `CLAUDE.md` were independently fetched from
   the same commit and diffed — **all seven files byte-for-byte identical**.

## Confidence and limitations

**Very high confidence.** Seven independent, unrelated files matched
byte-for-byte, and approximately 15 independently-varying dependency-version
pins all converge on the exact same single-commit window with no gaps or
contradictions — this combination is not plausibly coincidental. The
determination was made by direct content comparison against fetched
upstream source, not by version-number similarity alone (per the task's
explicit instruction not to select a commit based on similarity only).

**Limitation**: this identifies the commit the VENDORED FILES were copied
from; it does not itself prove the exact moment or process by which the
copy was made (e.g., whether via `git archive`, a manual download, or an
editable `pip install -e` against a locally-cloned checkout that was later
deleted) — that procedural detail remains unrecoverable and is not needed
for reproducibility purposes (the file content itself is what matters, and
that is now pinned exactly).

## What this changes in the prior audits

Both `chapter3_methodology_audit/chapter3_source_of_truth.md` (§A) and
`analysis/canonical/chapter3_final_source_of_truth.md`
(§1) state the upstream commit as `UNKNOWN`. That determination is now
**superseded** by this finding: the exact upstream commit is
`1f24f1142db24f81e29eb88751985a00ec8ed580` (2026-06-19). The local diff
against this commit (rather than against the repo's own first-commit
snapshot, which already includes whatever the vendoring process may have
altered) should be used for any future "what changed vs. upstream" table.
