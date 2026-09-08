# Curation and splitting

Finding near-duplicates, and dividing a corpus so that the evaluation means
something.

```bash
dslm curate --corpus corpus.jsonl.gz --out clean.jsonl.gz --card DATA-CARD.md
dslm split  --corpus clean.jsonl.gz  --out-dir splits/
```

## Order matters

**Normalise, then compare, then split.** Comparing before normalising makes the
deduplicator partly a case-sensitivity detector, and every count downstream is
off by an amount nobody can quantify afterwards.

Normalisation for *comparison* lowercases, composes to NFC, strips punctuation
and collapses whitespace. The stored text keeps what the technician actually
wrote — applying the comparison form to it would train the model on something
nobody typed.

One exception: hyphens and slashes survive. `32-41-00` is a manual reference and
`p/n` is a part number, and a pre-tokeniser that discards them destroys the very
structure a domain tokenizer exists to learn.

## Near-duplicates, not exact ones

A hash set finds byte-identical records, which in operational text is the easy
and rare case. The expensive case is the same write-up with the station code
changed:

```
A320 LHR: main gear — shock strut pressure low at the gate. serviced ...
A320 AMS: main gear — shock strut pressure low at the gate. serviced ...
```

For training purposes that is one record. For a hash set it is two.

## The parameters, derived rather than copied

This is the part worth reading, because the conventional setting is wrong here
and the arithmetic says exactly why.

A record of `n` words yields `S = n − w + 1` shingles. Changing one word
destroys at most `w` of them on each side, so the **best achievable similarity**
between a record and its one-word variant is:

```
J = (S − w) / (S + w)
```

For a 21-word maintenance record:

| Shingle width | Best achievable J | Conventional threshold |
| --- | --- | --- |
| 3 | 0.78 | — |
| 4 | 0.70 | — |
| **5** | **0.63** | **0.8** |

**The conventional 5-word shingle at 0.8 cannot detect a one-word edit at all on
records this short.** It reports that a corpus full of near-duplicates contains
none. The setting comes from web-scale document deduplication, where a document
is hundreds of words and a one-word edit is a rounding error.

The formula predicts the measured median to three decimal places at every width,
and `tests/unit/test_curate.py::TestTheShingleArithmetic` asserts it.

### The threshold, chosen by measurement

At a 3-word shingle, measured over the shipped corpus:

| Population | Score |
| --- | --- |
| a planted one-word variant | **0.727** and above (median 0.778) |
| an unrelated pair | **0.471** and below (99.9th percentile 0.306) |

0.6 sits in the gap with margin on both sides.

### The banding has to match the threshold

MinHash with LSH banding: two records are compared exactly only if they agree on
a whole band of the signature. The probability of that is
`1 − (1 − J^rows)^bands`, an S-curve whose midpoint is `(1/bands)^(1/rows)`.

| Bands × rows | Midpoint | P at J=0.6 | P at J=0.8 |
| --- | --- | --- | --- |
| 16 × 8 | 0.707 | 0.237 | 0.947 |
| **32 × 4** | **0.420** | **0.988** | **1.000** |

16 × 8 is the textbook default and was this project's first setting. With a 0.6
threshold it finds a pair sitting *at* that threshold barely one time in four —
a deduplicator missing three quarters of exactly the cases it was configured to
catch. Recall over the planted duplicates was **0.950**.

At 32 × 4 the midpoint sits below the threshold and recall is **1.000**. The
cost is more candidate pairs to verify exactly — 41,000 rather than 5,600 on the
shipped corpus, against the 18 million that comparing every pair would need.
Banding exists to avoid that, not to be as cheap as possible.

A unit test caught this.

## Curation reports; removal is separate

`find_duplicates` returns clusters and changes nothing. `apply_curation` removes
them. The CLI writes a deduplicated corpus only when `--out` is given.

A curation step that silently removed a quarter of a corpus would be the most
consequential unlogged action in the pipeline.

```
417 of 6000 record(s) are duplicates (7.0%) in 402 cluster(s); 180 exact, 237 near
```

The kept record is the first by id — arbitrary, and deliberately so. Any rule
preferring the longest text would be a silent editorial decision about which
write-up is the real one.

## The data card is generated

Never typed. A hand-written card is correct on the day it is written and wrong
from the first change afterwards, which is worse than none because it is
believed.

```bash
dslm curate --corpus corpus.jsonl.gz --card DATA-CARD.md
```

It reports the record count and digest, every source and licence present, the
label distribution with the majority share (the accuracy of guessing), text
length statistics, and the duplicate clusters.

`source` and `license` are **required fields on every record**, not a note in a
README. A corpus whose licence cannot be established cannot be published, and by
the time anyone asks, the person who assembled it has usually left.

## Splitting without leaking

Three failures, in the order they are usually made.

**Splitting before deduplicating.** A record and its near-duplicate land on
opposite sides; the model memorises one and is tested on the other.

**Splitting randomly when the data has groups.** If the same aircraft, shop or
author appears on both sides, the model can key on the group. `--group-by <key>`
is available and off by default, because whether a group leaks is a property of
the domain rather than of the code.

**Splitting non-deterministically.** A split that differs between runs makes
every comparison between two models meaningless. The seed is required and is
part of the split's digest.

So `split_corpus` takes the duplicate clusters and keeps every member on one
side, merging transitively with the optional grouping key — if A duplicates B
and B shares an aircraft with C, all three stay together.

```
5583 record(s): train 4467, dev 558, test 558 (seed 1)
```

Groups are shuffled and filled greedily into whichever part is furthest below
its target share. Greedy rather than proportional because groups have different
sizes, and the part that ends up short under a proportional rule is always test.

### Dev exists so that test is not touched

A repository with no dev split has either not tuned anything or has tuned on
test. Every hyperparameter in this project — the weight decay, the epoch count,
the vocabulary size, the early-stopping point — was chosen on dev.

### And none of it is sufficient

Grouping is not a proof. The check that actually catches the mistake is the
[contamination gate](contamination.md), which measures the splits this module
produced rather than trusting that it did its job.

## See also

- [Contamination](contamination.md) — measuring what this tried to prevent
- [ADR-005](../ARCHITECTURE.md) — the shingle and banding decisions
- `python examples/quickstart.py` — the pipeline, executing
