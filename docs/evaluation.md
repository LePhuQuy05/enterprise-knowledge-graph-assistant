# Evaluation and limitations

The project measures retrieval, source grounding, failure handling and operational correctness separately. Passing tests does not establish answer accuracy or general attack resistance.

## Recorded retrieval experiment

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

## Security and operations

Offline fixtures cover authorization, cross-user access, malformed inputs, prompt-injection examples, corrupted snapshots, concurrency, ambiguous completion and budget accounting. A separate small live synthetic provider smoke and a 25-invocation model-free AWS operational sample were performed. These are bounded checks, not a penetration-test certification or capacity estimate.

The owner has reported successful Vietnamese answers with sources/history and successful account access after email verification. Those confirmations are distinguished from an independently captured browser JWT trace and human semantic review.

## Reproduction

The committed test suite recreates mechanics with synthetic fixtures. Corpus manifests and evaluation configuration identify the intended public sources and draft evidence. Exact replay of original scores additionally requires the original versioned documents, snapshots and frozen retrieval records, which are not distributed in Git.

The dashboard projection presents recorded diagnostics with source attribution and draft labels. Future experiments must preserve the source manifest, model/prompt versions, split boundaries, context/token budget, controls and cost receipts.

Remaining work includes independent human labels, a held-out benchmark, broader multilingual/source-relevance checks, semantic citation review, cloud adversarial/load/recovery testing and fair graph ablations. Publishing this repository does not mark those work items complete.
