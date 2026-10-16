# Retrieval-equivalence audit summary

Pair-rows attempted (example present under both condition and baseline): 64800
Pairs with both traces readable and a first-search event found: 63314

| Check | % identical (of readable pairs) |
|---|---|
| First search query (raw) | 50.7% |
| First search query (normalized) | 50.9% |
| Retrieved document-ID set | 66.3% |
| Retrieved rank order | 50.7% |
| Original (pre-suffix) observation hash | 42.5% |
| Shared-prefix hash | 42.5% |
| BM25 index hash | 100.0% |
| Retriever config hash | 100.0% |

Different query but identical retrieved document set: 12360 (19.5% of readable pairs)
Different retrieved evidence (document-ID sets differ): 21351 (33.7% of readable pairs)

**Interpretation guardrail:** this audit measures what the production traces actually contain. It does NOT assume identical retrieval across conditions -- see chapter3_source_of_truth.md section 7 and chapter3_final_source_of_truth.md section 11 for why no structural guarantee exists in production (`run_paired_replay` is not the production entry point).
