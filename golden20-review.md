# Golden-20 review and labeling guide

This file is the human side of the golden-20 evaluation. The code assigns,
the reviewer confirms. Nothing here runs by itself.

## 1. Category review (before any run)

`scripts/build_golden_set.py` assigns each picked chunk a `golden_category`
(clean, dense, messy, boilerplate) with a `golden_category_reason` naming the
exact rule that fired, for example `is_table is true` or
`est_tokens 1203 >= 800`.

The categories are heuristic starting points, not certified truth:

- `dense` is a size rule (est_tokens >= 800). It is a proxy for
  "dense/multi-hop"; only reading the chunk confirms it really needs
  multi-entity, multi-relation reasoning.
- `boilerplate` is a marker rule (safe-harbor or cover-page wording). Noise
  inside a real filing can still name valid ontology entities, for example a
  forward-looking paragraph that names the filing company and a risk.

Review pass: open the generated `data/golden/golden20-chunks-<date>.json`,
read each chunk, and fix any `golden_category` that does not match what the
text really is. The mix targets (8 clean, 6 dense, 4 messy, 2 boilerplate)
describe the intended benchmark shape; an honest relabel beats a forced mix.

## 2. Labeling rules (ground truth)

Labels follow the same shape as `tests/fixtures/labels-positive-sample.json`:
reviewed flag, reviewer line, entities, relationships, name groups.

- Every label must be grounded in the frozen ontology
  (`configs/ontology.json`: 7 entity types, 9 relation types) and in the
  chunk's own text. If a name is not in the text, it is not a label.
- Evidence quotes are verbatim substrings of the chunk.
- Aliases go together only when the source defines the same identity
  ("the Company" = Apple Inc. inside one filing).
- Boilerplate/noise chunks: label what is truly there. If review finds
  extractable entities, record them. An empty expected output is only
  correct when review finds genuinely nothing extractable.
- The checklist target "0 entities from the 2 noise chunks" therefore
  applies only when the reviewed labels are genuinely empty. If a noise
  chunk legitimately contains entities, the labels say so and the pass/fail
  threshold is re-set from the labels, never the other way round.

Hand-label the planned 5-8 chunks first (the benchmark's minimum), and keep
the label file next to the golden set, for example
`data/golden/golden20-labels-<date>.json`. New file each revision; never
overwrite.

## 3. Few-shot separation

The prompt's worked examples live in
`configs/golden_fewshot_examples.json` (three Apple 10-K 2023 excerpts with
agent-reviewed labels; `user_confirmed` is false until Tanya confirms them).

`build_golden_set.py` keeps them out of the evaluation set twice: by chunk id
and by text overlap (a candidate sharing any real span of an example's text
is excluded and counted under `excluded_by_example_text_overlap`). The
examples and the golden set stay disjoint, so nothing the prompt shows the
model can leak into scoring.

## 4. Verification status, stated plainly

- Offline-tested (fake provider, no network): prompt assembly, metrics
  honesty rules (missing fields become "unavailable" and fail verification),
  429 stop behavior, chunk limit, curation rules, few-shot schema shape.
  446 repo tests pass including 17 new ones.
- Not yet done: any real OpenRouter call. Whether a free model returns every
  usage field (cached tokens, reasoning tokens, cost) is only knowable from
  a real response; the 2-chunk smoke run is exactly that check.
- Schema pass means shape, ontology and provenance checks passed. It is not
  an accuracy claim. Accuracy comes from scoring run outputs against the
  reviewed labels.
- Free vs free cannot show a dollar saving: both models report cost 0.
  Compare tokens, latency, schema pass and label quality instead.
