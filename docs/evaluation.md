# Evaluation and limitations

The project measures retrieval, source grounding, failure handling and operational correctness separately. Passing tests does not establish answer accuracy or general attack resistance.

## Expanded development experiment — 20 October 2026

The expansion contains **50 scenarios with English/Vietnamese versions**, forming 100 language inputs: 22 factual, 10 entity/relation, 8 candidate multi-hop and 10 corpus-unanswerable scenarios. Forty answerable scenarios belong to 23 intent groups; all 50 belong to 31 groups. Translations and related questions share groups. The 800 question/system rows are repeated measurements, not 800 independent questions.

| Configuration | English CompleteEvidence@5 | Vietnamese CompleteEvidence@5 |
|---|---:|---:|
| BM25 | 34/40 (85.0%) | 17/40 (42.5%) |
| BGE vector | 33/40 (82.5%) | 21/40 (52.5%) |
| Hybrid, RRF60 | 37/40 (92.5%) | 25/40 (62.5%) |
| Mixed graph 1-hop | 37/40 (92.5%) | 24/40 (60.0%) |
| Mixed graph 2-hop | 37/40 (92.5%) | 24/40 (60.0%) |
| Mention-only expansion | 37/40 (92.5%) | 24/40 (60.0%) |
| Relation-only 1-hop | 37/40 (92.5%) | 24/40 (60.0%) |
| Relation-only 2-hop | 37/40 (92.5%) | 24/40 (60.0%) |

**These scores use AI-draft labels; the human-approved denominator is zero.** The set is development data, not an established public benchmark or independent holdout. The older 15 scenarios remain unchanged within the expansion, so those examples are exposed regressions. Do not compare 37/40 directly with historical 10/12 as a longitudinal improvement.

### Data credibility and comparison controls

- All 20 source URLs are on `docs.aws.amazon.com`. The frozen snapshot contains 236 chunks. Gold is anchored to document versions, exact normalized-text offsets and quotes, rather than retrieved chunk IDs. All 53 required evidence groups match frozen source text and are recoverable from the entire corpus. This establishes source integrity/recoverability, **not semantic label correctness or exhaustive alternative evidence**.
- See [English cases](../configs/evaluation/expanded-development-en-v1.json), [Vietnamese cases](../configs/evaluation/expanded-development-vi-v1.json), and [source index](../benchmarks/expanded-development-v1/source-index.json) for URLs, fetch times, raw/normalized hashes, normalization versions and chunk ranges. Web pages may subsequently change; original frozen files are needed for full text verification.
- The five-system run freezes data/model/code/lock hashes before observing its rankings. All modes use identical documents, source chunks, BGE384 query vectors, top5 scoring, RRF60, graph resource bounds and the same context packer (up to eight chunks / 4096 proxy tokens). Gold facts, categories and reference answers never enter retrieval. Baseline and graph snapshots are checked for identical documents, chunks, recipe and vectors.
- Raw Vietnamese is measured without translating it to English. BGE-small-en-v1.5 is an English model, and the current lexical tokenizer is ASCII-oriented. Poor Vietnamese scores identify a real limitation in this configuration; they are not the success rate of the separate live-search query-rewrite path. The model language is documented in the [BAAI model card](https://huggingface.co/BAAI/bge-small-en-v1.5).
- Graph holds only 14 experimental assertions from six chunks; seven chunks were accepted by extraction, including one with no relation assertions. Mixed graph reports candidates for 12/50 English and 17/50 Vietnamese inputs, but packs zero assertions. A follow-up removes mention expansion while preserving relation traversal, or removes relation adjacency while preserving mentions, with seeds and budgets unchanged. Relation-only modes retain assertion evidence in 12/50 English and 17/50 Vietnamese contexts, yet show no completeness gain. This follow-up was designed after observing the mixed result and is explicitly exploratory development work.
- The [main summary](../benchmarks/expanded-development-v1/summary.json) reports per-language/category scores, context completeness, intent-group macro scores, graph fallback counts, paired wins/losses/ties, and 2,000 whole-intent-group bootstrap resamples (seed 20261020). Hybrid beats vector by 4/40 items, or 10 percentage points, on this English draft set. The conditional paired interval for vector-minus-hybrid includes zero; no statistical superiority or population/general-AWS claim is made. One local timing sample per input/system is diagnostic, not a latency SLO.

### Recompute the published numbers

```sh
uv sync --frozen
uv run python -m scripts.export_expanded_evaluation --audit
```

This checks **500 main rows and 300 graph-component rows** using committed source ranges and labels. It requires no credentials, downloaded model, paid inference or original corpus. It detects changed inputs/metrics and recomputes summaries. Checksums detect accidental changes; they are not an independent signature, human review, proof of source text, or scientific validation. No original AWS document corpus or model weights are distributed in these receipts.

For a fresh retrieval run, first prepare the exact original BGE model and hash-verified source snapshots using the local development workflow, then run `python -m scripts.evaluate_expanded`. Replay the resulting directory with `--replay`. A different source snapshot must be a new data/experiment version; do not silently adjust labels to fit current search output.

### What remains before a recruiter quality claim

1. Human review of source support, necessary conditions, alternative spans, translation equivalence, grouping and answerability. Multi-hop candidates require semantic necessity review; two source documents alone do not prove multi-hop reasoning.
2. Review answer correctness and whether each citation supports the full claim, separately from exact citation integrity. This separation follows the evaluation concerns in the [ALCE paper](https://arxiv.org/abs/2305.14627).
3. Freeze a reviewed configuration, then author and lock distinct holdout intent groups. Do not rename these development examples or their translations as holdout. Group-based split semantics are explained by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html).

The current evidence supports an engineering statement: **built and audited an eight-configuration bilingual development evaluation, identified a multilingual retrieval gap, and retained negative graph results**. It does not support “92.5% answer accuracy across AWS.”

## Historical retrieval experiment

The pilot snapshot contains 20 documents and 236 chunks. The development dataset contains 15 English questions: 12 answerable and three corpus-unanswerable. Evidence labels remain `AI_DRAFT`; no human-approved score is claimed.

| System | CompleteEvidence@5 |
|---|---:|
| Vector | 8/12 |
| Hybrid: vector + BM25 with rank fusion | 10/12 |
| Draft one-hop/two-hop graph expansion | 10/12 |
| Service-mention graph ablation | 8/12 |

CompleteEvidence@5 asks whether all required evidence groups are covered by the first five retrieved chunks. It is not a measure of generated-answer accuracy. Unanswerable questions are excluded from that denominator and require separate abstention evaluation.

The graph variants did not demonstrate improvement over hybrid. Graph answering therefore remains disabled in serving; negative results are retained as an engineering decision rather than presented as a benefit.

## Answer grounding

Generation uses structured output and server-selected evidence spans. Citation integrity checks reject unknown source identifiers, mismatched versions and unsupported quote ranges. They do not prove that every sentence follows logically from a genuine quote.

The broader bilingual release evaluation has not met its full quality gate. Observed failure modes include irrelevant evidence, incomplete multi-service coverage, invalid selected evidence identifiers and structurally valid but unsupported claims. The interface must preserve partial/abstained/error states rather than repair a result through unbounded paid retries.

### Expanded paired-language answer experiment

The 20 October experiment generated **100 responses for the 50 paired scenarios**, using the serving V9 generation recipe and the same frozen English hybrid context for both languages. It isolates bilingual generation; it does not evaluate HTTP, authentication, query rewriting, live search or answer generation for all eight retrievers.

| Automated observation | English | Vietnamese |
|---|---:|---:|
| Recorded generation attempts | 50 | 50 |
| Strict schema/source-span integrity pass | 50/50 | 48/50 |
| Corpus-unanswerable returned ABSTAINED | 10/10 | 9/10 |
| Human semantic correctness / citation entailment | Not scored | Not scored |

The two Vietnamese failures omit the mandatory `reason` field; both paid raw outputs remain recorded. One missing-account-state control returns PARTIAL with general encryption facts, not the requested actual key ARN; it requires semantic review and is not counted as a strict abstention pass. **98/100 structural passes is not 98% answer accuracy.**

All 100 operations settled; their published-peak usage cost upper bound is **USD 0.124880**, not a reconciled provider invoice. Six canary calls plus 94 subsequent calls were made, without retrying failed samples. Replay makes zero new calls. The experiment uses a separate USD 0.50 managed run under the unchanged shared USD 2 external allowance; previous spending and reservations are retained.

See the [answer plan](../benchmarks/expanded-development-v1/answer-manifest.json), [summary](../benchmarks/expanded-development-v1/answer-summary.json), and [per-case responses/failures](../benchmarks/expanded-development-v1/answers.jsonl). Human semantic review and independent holdout remain pending; no automatic LLM judge score is substituted for them.

## Security and operations

Offline fixtures cover authorization, cross-user access, malformed inputs, prompt-injection examples, corrupted snapshots, concurrency, ambiguous completion and budget accounting. A separate small live synthetic provider smoke and a 25-invocation model-free AWS operational sample were performed. These are bounded checks, not a penetration-test certification or capacity estimate.

The owner has reported successful Vietnamese answers with sources/history and successful account access after email verification. Those confirmations are distinguished from an independently captured browser JWT trace and human semantic review.

## Reproduction

The committed test suite recreates mechanics with synthetic fixtures. Corpus manifests and evaluation configuration identify the intended public sources and draft evidence. Exact replay of original scores additionally requires the original versioned documents, snapshots and frozen retrieval records, which are not distributed in Git.

The dashboard projection presents recorded diagnostics with source attribution and draft labels. Future experiments must preserve the source manifest, model/prompt versions, split boundaries, context/token budget, controls and cost receipts.

Remaining work includes independent human labels, a held-out benchmark, broader multilingual/source-relevance checks, semantic citation review, cloud adversarial/load/recovery testing and fair graph ablations. Publishing this repository does not mark those work items complete.
