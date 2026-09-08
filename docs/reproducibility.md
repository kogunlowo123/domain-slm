# Reproducibility

What is bit-identical, what is not, and the measurements that decided which.

This project wanted one rule: every artefact checked by content address. That
was measured, and for one of them it turned out to be impossible. This page is
the measurement.

## The short version

| Artefact | Made of | Reproducible? | Gate |
| --- | --- | --- | --- |
| Corpus | integers and strings | **exactly**, everywhere | digest |
| Split | integers and strings | **exactly**, everywhere | digest |
| Tokenizer | integers and strings | **exactly**, everywhere | digest |
| Naive Bayes | counts, then `log` | to within libm | metrics |
| **Neural weights** | float64 through BLAS | **no** | metrics within a tolerance |
| Reported metrics | float64 | bit-identical at this scale | — |

## The measurement

Same seed, same pinned NumPy, Windows against Linux, after 30 training steps:

- **the loss is bit-identical** — `0x1.8edded8f6b1e1p+2` on both;
- **1,344 of 55,872 weights differ**, by at most **2.4e-14 relative**.

Accumulation order inside a matrix multiply is a property of the BLAS build, and
no seed controls it. There is no flag that fixes this and no version pin that
avoids it.

### The one-ULP finding on the way there

Before the training loop was even reached, the *initialisation* differed.
Bisecting it:

- `default_rng(seed).integers(...)` — bit-identical.
- `default_rng(seed).random(...)` — bit-identical.
- `default_rng(seed).standard_normal(...)` — **one value in 32,768 differed, by
  one unit in the last place.**

`standard_normal` uses a ziggurat, whose rare tail branch falls back to a
computation involving `log`. libm's `log` differs by one ULP between the two
platforms. One bit, in one weight — which then propagates through every
subsequent weight of training.

**So initialisation is uniform, not normal.** Glorot-uniform is standard
practice anyway, and `random()` is integer-to-float only with no libm call. That
made initialisation bit-identical on both platforms.

It did not fix training, which is what the measurement above records. The
divergence moved from the initialiser to the BLAS.

### Drawing flat and reshaping

Weights are drawn as one flat vector and reshaped:

```python
flat = rng.random(fan_in * fan_out) * 2.0 - 1.0
return (flat * limit).reshape(fan_in, fan_out)
```

So the number of variates consumed depends only on the parameter count, never on
the shape. Changing a layer's shape without changing its size then cannot
silently reshuffle every subsequent draw.

## What follows from it

**Exact digests for the integer artefacts.** `dslm check` regenerates a corpus
from its plan and compares digests — over what a model learns from, not over the
bytes, so reordering a file or adding a key to its metadata does not move it.

**Metrics within a tolerance for the model.** The regression gate allows a drop
of 0.005. That tolerance is not absorbing floating-point noise — the metrics are
bit-identical at this scale — it absorbs the genuine run-to-run variation a
different NumPy resolving a different BLAS produces.

**Same-platform reproducibility is still exact**, and is asserted:
`test_the_same_seed_gives_the_same_model_on_this_machine` compares weight
digests. Cross-platform equality is deliberately *not* asserted, because it
would fail in CI for a reason having nothing to do with the code.

## What is reproducible, precisely

### The corpus

```bash
dslm synth --plan main --out corpus.jsonl.gz
dslm check --plan main --corpus corpus.jsonl.gz
```

`random.Random` rather than NumPy's generator: every choice is among strings, so
the stdlib Mersenne Twister is exactly reproducible with none of the divergence
above. The start of the corpus is fixed, not `now()` — regenerating tomorrow
must produce the same file.

### Gzip

Windows are written with a **zero timestamp** in the gzip member. The default
stamps the current time, so the same corpus written twice would produce
different bytes and every regenerated file would differ from the committed one
for a reason no reader could see — turning the drift check into noise.

`np.savez` rather than `savez_compressed` for the same reason: the weights are
dense floats that compress by a few percent, which is not worth making the
artefact's bytes depend on the zlib version.

### The tokenizer

Ties in pair frequency are broken by the pair itself, sorted, so a dictionary's
iteration order can never decide a merge. The digest is stored in the file and
rechecked on load.

### MinHash

The permutation family is derived from a fixed seed with blake2b, and shingles
are hashed with blake2b rather than Python's `hash` — which is salted per
process, so a signature computed today would not match one computed tomorrow and
every committed report would be irreproducible.

## Reproducing the platform measurement yourself

The probe is small enough to paste. Train the shipped architecture for 30 steps
on each platform, print the loss as a hex float and a digest of the weights, and
compare. On this project's two platforms the loss matched exactly and the
digests did not.

Run the same thing twice on one machine and both match — which is the point:
the divergence is between platforms, not between runs.

## What this means for you

If you are gating on a model artefact's bytes, **stop**. Gate on what the model
*does*, with a tolerance chosen against a measurement of your own stack, and
digest the things that genuinely are deterministic — the data, the split, the
vocabulary.

Claiming a byte-exact model across platforms would be a claim this project
measured and found to be false.

## See also

- [ADR-003](../ARCHITECTURE.md) — the decision
- [The model](model.md) — what the tolerance is applied to
- [CI](ci.md) — where the gates run
