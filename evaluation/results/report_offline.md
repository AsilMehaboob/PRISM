# PRISM evaluation report

- Backend: **offline (lexical fallback for Stage 2 and Stage 5; no network)**
- dev split: 57 benign / 41 attack. heldout split: 28 benign / 28 attack (hand-written; see `evaluation/dataset.py`). Tune on dev only; quote heldout.
- Attack success = payload persisted in LONGTERM. Brackets are Wilson 95% CIs.
- A rejected benign item is stored as SCRATCH by `memory/router.py`, so it counts as a miss for its true tier.

## Main comparison (threshold 0.65)

| Config | Split | Benign acc | Macro-F1 | False-reject | ASR (LONGTERM) | ASR (any persistent tier) | mean ms |
|---|---|---|---|---|---|---|---|
| A flat memory (baseline) | dev |  38.6% (22/57) [27-52] | 0.19 |   0.0% (0/57) [0-6] | 100.0% (41/41) [91-100] | 100.0% (41/41) [91-100] | 0.0 |
| A flat memory (baseline) | heldout |  42.9% (12/28) [27-61] | 0.20 |   0.0% (0/28) [0-12] | 100.0% (28/28) [88-100] | 100.0% (28/28) [88-100] | 0.0 |
| B stage2 only | dev |  89.5% (51/57) [79-95] | 0.90 |   0.0% (0/57) [0-6] |  92.7% (38/41) [81-97] |  92.7% (38/41) [81-97] | 0.0 |
| B stage2 only | heldout |  85.7% (24/28) [69-94] | 0.84 |   0.0% (0/28) [0-12] |  89.3% (25/28) [73-96] |  92.9% (26/28) [77-98] | 0.0 |
| C stage1+2 | dev |  87.7% (50/57) [77-94] | 0.87 |   1.8% (1/57) [0-9] |  90.2% (37/41) [77-96] |  90.2% (37/41) [77-96] | 0.1 |
| C stage1+2 | heldout |  82.1% (23/28) [64-92] | 0.78 |   3.6% (1/28) [1-18] |  89.3% (25/28) [73-96] |  92.9% (26/28) [77-98] | 0.1 |
| D stage1+2+4 (no LLM judge) | dev |  87.7% (50/57) [77-94] | 0.87 |   1.8% (1/57) [0-9] |  82.9% (34/41) [69-91] |  82.9% (34/41) [69-91] | 0.1 |
| D stage1+2+4 (no LLM judge) | heldout |  82.1% (23/28) [64-92] | 0.78 |   3.6% (1/28) [1-18] |  82.1% (23/28) [64-92] |  85.7% (24/28) [69-94] | 0.1 |
| E1 full pipeline, no trust cap (legacy) | dev |  87.7% (50/57) [77-94] | 0.87 |   1.8% (1/57) [0-9] |  82.9% (34/41) [69-91] |  82.9% (34/41) [69-91] | 0.1 |
| E1 full pipeline, no trust cap (legacy) | heldout |  82.1% (23/28) [64-92] | 0.78 |   3.6% (1/28) [1-18] |  82.1% (23/28) [64-92] |  85.7% (24/28) [69-94] | 0.1 |
| F full pipeline | dev |  87.7% (50/57) [77-94] | 0.87 |   1.8% (1/57) [0-9] |  80.5% (33/41) [66-90] |  82.9% (34/41) [69-91] | 0.1 |
| F full pipeline | heldout |  82.1% (23/28) [64-92] | 0.78 |   3.6% (1/28) [1-18] |  78.6% (22/28) [60-90] |  85.7% (24/28) [69-94] | 0.1 |

Ablation fidelity: re-composed full pipeline vs `classifier.pipeline.run` differ on **0/154** items (expected 0).

## Attack success by category (dev + heldout)

| Category | F full pipeline | Flat |
|---|---|---|
| direct_injection | 100.0% (10/10) [72-100] | 100% |
| preference_hijack | 100.0% (10/10) [72-100] | 100% |
| false_fact | 100.0% (10/10) [72-100] | 100% |
| exfiltration |  85.7% (6/7) [49-97] | 100% |
| paraphrased |  90.0% (9/10) [60-98] | 100% |
| obfuscated | 100.0% (10/10) [72-100] | 100% |
| tool_injection |   0.0% (0/7) [0-35] | 100% |
| untrusted_source |   0.0% (0/5) [-0-43] | 100% |

## Benign confusion matrix, full pipeline (true->predicted: count)

- `LONGTERM->LONGTERM`: 34
- `SCRATCH->SCRATCH`: 24
- `SESSION->LONGTERM`: 10
- `SESSION->SCRATCH`: 2
- `SESSION->SESSION`: 15

Per-tier F1: SCRATCH=0.96, SESSION=0.71, LONGTERM=0.87

## Threshold sweep (full pipeline, dev + heldout)

| Threshold | Sent to LLM judge | Benign acc | ASR (LONGTERM) |
|---|---|---|---|
| 0.40 | 7% |  85.9% (73/85) [77-92] |  79.7% (55/69) [69-88] |
| 0.50 | 7% |  85.9% (73/85) [77-92] |  79.7% (55/69) [69-88] |
| 0.60 | 7% |  85.9% (73/85) [77-92] |  79.7% (55/69) [69-88] |
| 0.65 | 33% |  85.9% (73/85) [77-92] |  79.7% (55/69) [69-88] |
| 0.70 | 33% |  85.9% (73/85) [77-92] |  79.7% (55/69) [69-88] |
| 0.80 | 42% |  85.9% (73/85) [77-92] |  79.7% (55/69) [69-88] |
| 0.90 | 42% |  85.9% (73/85) [77-92] |  79.7% (55/69) [69-88] |

## Integrity / tier-isolation tests (deterministic)

| Test | Result |
|---|---|
| signed item verifies | PASS |
| tampered content is rejected | PASS |
| tampered timestamp is rejected | PASS |
| re-labelled id is rejected | PASS |
| signature from a different key is rejected | PASS |
| signature replay onto other content is rejected | PASS |
| malformed signature is rejected (no exception) | PASS |
| store refuses unsigned long-term item | PASS |
| store refuses item signed by wrong key | PASS |
| store refuses non-LONGTERM tier | PASS |
| store refuses a validly signed item from untrusted source 'tool' | PASS |
| store refuses a validly signed item from untrusted source 'agent' | PASS |
| store refuses a validly signed item from untrusted source 'web' | PASS |
| store accepts a validly signed item | PASS |
| retrieval drops rows injected straight into the DB | PASS |
| retrieval drops a stored row edited after signing | PASS |
| session checksum detects accidental corruption | PASS |
| session tier resists deliberate forgery | FAIL - LIMITATION: Adler-32 is unkeyed; anyone can forge it. Session tier relies on TTL, not integrity. |
