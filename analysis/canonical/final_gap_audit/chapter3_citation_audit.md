# Chapter 3 citation audit (Phase 9)

**No `references.bib` exists anywhere on this machine** (confirmed in
`thesis_project_inventory.md` — a full recursive search of
`/home/sbhengesa` found zero `.bib` files). This audit therefore cannot
report "existing entries" or check them against primary sources, since
there is no existing bibliography to check. Instead, this file reports the
**verified primary-source bibliographic data** for every item Chapter 3 is
expected to cite, obtained via live web search/fetch against arXiv, ACL
Anthology, MIT Press, and the official Meta/Google/GitHub sources
(not fabricated, not from training-data memory alone) — this is exactly the
data needed to populate `chapter3_missing_citations.bib` (all entries there
are "missing" by definition, since no bib file exists to compare against).

| Item | Verified? | Title | Authors | Year | Venue | DOI/arXiv | URL |
|---|---|---|---|---|---|---|---|
| Open Deep Research | Yes (repository, no paper) | Open Deep Research (software repository) | LangChain, Inc. | 2025 (per local `LICENSE` copyright year) | GitHub repository | n/a (no DOI/arXiv — it is software, not a paper) | https://github.com/langchain-ai/open_deep_research |
| LangGraph | Partial | No citable peer-reviewed primary paper found | LangChain, Inc. | n/a | Official documentation/repository only | n/a | https://github.com/langchain-ai/langgraph ; https://reference.langchain.com/python/langgraph |
| HotpotQA | Yes | HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering | Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, Christopher D. Manning | 2018 | EMNLP 2018 | arXiv:1809.09600 | https://arxiv.org/abs/1809.09600 |
| 2WikiMultiHopQA | Yes | Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps | Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, Akiko Aizawa | 2020 | COLING 2020 | arXiv:2011.01060 | https://arxiv.org/abs/2011.01060 |
| MuSiQue | Yes | MuSiQue: Multihop Questions via Single-hop Question Composition | Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, Ashish Sabharwal | 2022 | Transactions of the ACL (TACL), Vol. 10, pp. 539-554 | DOI:10.1162/tacl_a_00475 / arXiv:2108.00573 | https://doi.org/10.1162/tacl_a_00475 |
| MultiHopRAG | Yes | MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries | Yixuan Tang, Yi Yang | 2024 | COLM 2024 | arXiv:2401.15391 | https://arxiv.org/abs/2401.15391 |
| Qwen2.5 | Yes | Qwen2.5 Technical Report | Qwen Team (An Yang et al.) | 2024 | arXiv preprint | arXiv:2412.15115 | https://arxiv.org/abs/2412.15115 |
| Llama 3.1 | Yes | The Llama 3 Herd of Models | Meta AI (Grattafiori et al.) | 2024 | arXiv preprint | arXiv:2407.21783 | https://arxiv.org/abs/2407.21783 |
| Gemma 3 | Yes | Gemma 3 Technical Report | Gemma Team, Google DeepMind | 2025 | arXiv preprint | arXiv:2503.19786 | https://arxiv.org/abs/2503.19786 |
| BM25 | Yes | The Probabilistic Relevance Framework: BM25 and Beyond | Stephen Robertson, Hugo Zaragoza | 2009 | Foundations and Trends in Information Retrieval, Vol. 3, No. 4 | DOI:10.1561/1500000019 | https://doi.org/10.1561/1500000019 |

## Per-entry notes

- **Open Deep Research**: cite as a software repository (`@misc`/`@software`
  BibTeX entry), not a paper — none exists. The exact vendored commit is now
  known (`1f24f1142db24f81e29eb88751985a00ec8ed580`, see
  `upstream_odr_recovery.md`) and should be cited in the URL/note field for
  precision (e.g. `note = {commit 1f24f114, accessed 2026-09-11}`).
- **LangGraph**: no peer-reviewed primary source exists. Cite the official
  GitHub repository and/or documentation site as software, consistent with
  how most agentic-framework tooling (LangChain, LlamaIndex, etc.) is
  typically cited in ML systems papers when no paper exists.
- **MuSiQue**: note the paper's own unusual leading music-note character
  ("♫") in its title as rendered on ACL Anthology/TACL — this is a stylistic
  choice by the authors, not a typo; safe to omit in a BibTeX `title` field.
- **Qwen2.5/Llama 3.1/Gemma 3**: all three are verified against their
  official technical-report arXiv preprints (the standard citation practice
  for open-weight model releases); none of the three has been verified
  against a peer-reviewed venue beyond arXiv, since none has one at this
  writing.
- **BM25**: cited via the canonical, most complete methodological reference
  (Robertson & Zaragoza 2009) rather than the original 1994 TREC working
  notes, which is standard modern practice and gives a real DOI.

## What Chapter 3 must still cite that is NOT included above

The task also asks about "any evaluation method that Chapter 3 must cite" —
this audit did not attempt to identify every possible evaluation-method
citation (e.g., exact-match/F1 scoring conventions trace back to SQuAD,
Rajpurkar et al. 2016) because the actual Chapter 3 text (which would state
exactly which evaluation conventions it invokes) does not exist on this
machine to check against (see `thesis_project_inventory.md`). If Chapter 3
cites SQuAD-style EM/F1 scoring by name, that citation
(Rajpurkar et al., "SQuAD: 100,000+ Questions for Machine Comprehension of
Text", EMNLP 2016, arXiv:1606.05250) was NOT independently re-verified in
this pass and should be checked before use.
