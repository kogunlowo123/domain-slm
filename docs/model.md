# The model

What it is, what it scores, and why the linear control it is compared against
is the most useful thing in this document.

## The architecture

Token embeddings → mean pool → one hidden layer with `tanh` → softmax over 12
ATA chapters. About 55,000 parameters at the shipped size; it trains on 4,467
records in **five seconds** on one CPU core.

```
tokens ──▶ embedding (1024 × 48) ──▶ mean over real tokens
                                        │
                                        ▼
                                   W1 (48 × 96) + tanh
                                        │
                                        ▼
                                   W2 (96 × 12) ──▶ softmax
```

Mean pooling rather than attention, deliberately. A work order is twenty words
long and the task is essentially *"which vocabulary is this?"* — a bag of
learned embeddings has the capacity that needs. When a bag of *counts* is
already within a rounding error (below), adding attention would be answering a
question nobody asked.

Padding is masked out of the mean. Without that, a short record is mostly an
average of the padding embedding and the model's view of it depends on
`max_tokens`.

## The results

Measured on the shipped test split of 552 records:

| Model | Accuracy | Macro F1 | Parameters |
| --- | --- | --- | --- |
| majority class | 11.1% | — | 0 |
| **multinomial naive Bayes** | **92.03%** | 91.93% | **12,300** |
| neural | 92.39% | 92.28% | 55,020 |
| ceiling from label noise | 96.1% | — | — |

The ceiling is real and deliberate: 4% of the corpus is mislabelled on purpose,
because real maintenance data is, and a repository reporting an accuracy without
one is reporting against an imaginary target. The generator records which
records it corrupted, so the ceiling is computed rather than asserted.

## Is the difference real?

**No.** McNemar's exact test over the 14 records the two models disagree about:

```
naive-bayes and neural are statistically indistinguishable
(p = 0.791 over 14 disagreement(s); absence of evidence, not evidence of absence)
```

The wording matters. "Indistinguishable" and "the same" are different claims,
and only one of them is supported by a p-value.

### Why McNemar, and why the exact form

The two models are scored on **the same records**, so the samples are paired. A
two-proportion test assumes independence, throws away the pairing, and loses
most of its power. McNemar uses only the discordant records — the ones that
actually distinguish the models.

The exact binomial form rather than the chi-squared approximation, because the
approximation needs roughly 25 discordant pairs to be trustworthy and two
similar models on a few hundred records routinely produce fewer. Fourteen, here.

### What this means

**The linear control matches the neural model with a fifth of the parameters.**

That is the result, so it is on the front page. It disciplines everything else:
there is no case for attention, for more layers, or for a bigger embedding on
this task, and a repository that had not run the control could have claimed all
three.

If a change ever makes the neural model genuinely better,
`tests/integration/test_pipeline.py` fails and this page has to be rewritten —
which is the correct outcome.

## The hyperparameters, and the sweep that chose them

Every number below was chosen on the **dev** split. Test was not touched.

| Setting | Value | Why |
| --- | --- | --- |
| Vocabulary | 1024 | Larger did not help; see [tokenizer](tokenizer.md) |
| Embedding | 48 | The vocabulary is small and the distinctions are coarse |
| Hidden | 96 | — |
| Epochs | 20 | With early stopping on dev |
| Learning rate | 0.5 | Plain SGD; no momentum, no Adam |
| **Weight decay** | **1e-3** | Worth two full points. See below |

### More capacity makes it worse

The sweep, on dev:

| Embedding | Hidden | Epochs | Weight decay | Train acc | **Dev acc** |
| --- | --- | --- | --- | --- | --- |
| 48 | 96 | 12 | 1e-5 | 0.884 | 0.893 |
| 48 | 96 | 40 | 1e-5 | 0.942 | 0.869 |
| 64 | 128 | 40 | 1e-5 | 0.960 | 0.842 |
| 96 | 192 | 60 | 1e-5 | 0.9996 | **0.835** |
| 96 | 192 | 100 | 1e-5 | 1.0000 | 0.837 |

Training accuracy climbs to 1.0 while dev accuracy *falls* by six points. The
model has exactly enough capacity to memorise the corpus's 4% label noise, and
given the chance it does — it is learning the mistakes.

At `weight_decay=1e-3` with early stopping the same architecture reaches 0.918
on dev. That is the entire argument for the penalty, and it is measured rather
than assumed.

### Early stopping is real, not implied

`neural.train` keeps the parameters from the best **dev** epoch, not the last,
and reports which epoch it kept. Without a dev split it deliberately selects
nothing — selecting on training accuracy would choose the most overfitted model
available.

Selecting on *test* would be the standard mistake. The trainer is never given
the test split.

## The control

Multinomial naive Bayes over token counts. Forty lines, one hyperparameter,
trains in milliseconds.

Laplace smoothing at α = 1, and zero is refused: without smoothing a single
unseen token gives a class zero probability, so one novel word vetoes the class
a record obviously belongs to.

It is also the reproducible half of the pair — a sum of integer counts and one
logarithm per parameter, with no iterative optimisation. The neural model cannot
make that claim; see [reproducibility](reproducibility.md).

## What is reported besides accuracy

Accuracy is the number everybody reports and the one that hides the most.

**Macro F1**, because accuracy is dominated by the largest classes and a model
that gives up on the two rarest can still look fine.

**Per-class precision, recall and F1**, because a single macro number says
something is wrong and the table says what. The worst class is named in the
one-line summary: `worst class 36 at F1 0.864`.

**Mean confidence and top-decile accuracy.** A model right 92% of the time and
99% confident is differently wrong from one right 92% of the time that knows it.
If top-decile accuracy is not far above overall accuracy, the confidence carries
no information and should not be used to route or to abstain.

**The most-confused pairs.** Twelve classes is a 144-cell matrix nobody reads;
the five off-diagonal cells with the largest counts are the ones to act on.

## Persistence

A directory, not a file: `model.npz` for the weights, `model.json` for
everything a human or a report reads. Requiring an array library to answer
"which tokenizer was this trained against?" would guarantee nobody asks.

Three things are checked on load, each of which was a real failure before it was
a check: shapes must match the metadata, the tokenizer digest must match, and
the label space must be present. None of them can be caught by looking at the
output — a tokenizer mismatch is an accuracy several points lower that reads as
a bad model.

`np.savez`, never a pickle. Loading a pickle from an untrusted path executes
whatever it was told to.

## See also

- [Tokenizer](tokenizer.md) — what the model reads
- [Reproducibility](reproducibility.md) — what is bit-identical and what is not
- [ADR-002](../ARCHITECTURE.md) — the decision to always train both
