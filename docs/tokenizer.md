# Tokenizer

Byte-pair encoding, trained from scratch on the domain corpus.

```bash
dslm tokenizer --corpus splits/train.jsonl.gz --out tokenizer.json --vocab-size 1024
```

## Why train one

A general-purpose tokenizer is trained on general text and spends its vocabulary
accordingly. Domain text is not general: `actuator`, `pitot`, `bleed`, `strut`
and `amm 32-41-00` are common here and rare everywhere else, so a general
tokenizer shatters them into pieces. Every extra piece is another position the
model attends to and another parameter it spends rediscovering that three pieces
are one word.

The measurable version of that claim is **tokens per record**, so it is measured.

## The measurement

Two tokenizers of the **same vocabulary size**, differing only in what they were
trained on, both measured on the same domain corpus:

| Tokenizer | Tokens/record | Tokens/word | Chars/token |
| --- | --- | --- | --- |
| **domain** | **35.04** | **1.478** | **3.68** |
| general | 83.22 | 3.511 | 1.55 |

The general tokenizer needs **2.37× as many tokens per record**.

And on individual domain words:

| Word | Domain | General |
| --- | --- | --- |
| `actuator` | 1 piece | 7 pieces |
| `transponder` | 3 pieces | 7 pieces |
| `pressurisation` | 4 pieces | 10 pieces |
| `hydraulic` | 5 pieces | 9 pieces |

Run it: `python examples/tokenizer_demo.py`.

### A note on making that comparison fair

The first version of this demo used six sentences of sixty-five distinct words
as the "general" corpus. The general tokenizer could only make 455 of its 1,024
merges — it was losing partly because it had been **starved of vocabulary**, and
the headline ratio was 3.10× rather than 2.37×.

A comparison against a crippled baseline is not a comparison. The demo now uses
a synthesised general corpus large enough to fill the vocabulary, and prints the
unfilled-merge count for both sides so a reader can check that neither was
starved.

The general corpus is a stand-in, not a real one — that would be a licensing
question and a download — but it is genuinely out of domain, which is the only
property the comparison needs.

## What the implementation is

Word-level BPE over a whitespace pre-tokenisation, with a byte-level fallback.
The classic algorithm — count adjacent pair frequencies, merge the most
frequent, repeat — written out directly, because the point of this repository is
that every step is inspectable.

It is **not** a production tokenizer: no regex pre-tokeniser tuned over years,
no special-token machinery beyond what this project uses, and training on a
gigabyte would be slow. On this corpus it trains in half a second.

### Pre-tokenisation keeps domain punctuation

```python
pre_tokenise("amm 32-41-00 p/n 356-1194")
# ['amm', '32-41-00', 'p/n', '356-1194']
```

Deliberately *not* stripping punctuation: `32-41-00` is a manual reference and
`p/n` is a part number, and a pre-tokeniser that discards the hyphen and the
slash destroys the structure a domain tokenizer exists to learn.

### The end-of-word marker

`</w>` is appended to every word, so `valve` at the end of a word and `valve`
inside `valves` are different tokens. Without it the tokenizer cannot tell a
suffix from a whole word and merges across the boundary.

### It never emits an unknown token

Every one of the 256 bytes is in the vocabulary, so any piece that survives the
merges but is not in the vocabulary is emitted as its UTF-8 bytes.

```
domain      16 tokens, 0 unknown     # on "圧力 low strut p/n 356-1194"
```

A tokenizer that can produce `<unk>` loses information silently, and the loss
shows up as an accuracy the *model* gets blamed for.

## The fast path, and the reference it is checked against

The straightforward implementation recounts every adjacent pair in every word
after each merge. That took **twenty seconds** for 1,745 merges over 6,000
records — the slowest step in the pipeline by an order of magnitude, for a
tokenizer meant to be retrained casually.

A merge only changes the words containing the merged pair, so the counts only
need repairing there. With incremental counting the same run takes **0.56
seconds**: a 35× speedup.

`train_reference` — the slow, obvious version — is kept in the module, and
`tests/unit/test_tokenizer.py` asserts the two produce **identical merges** on
two different corpora. An optimisation whose only evidence is that it runs
faster is a rewrite nobody can trust.

## Determinism

Exactly reproducible, everywhere, forever. Every operation is over integers and
strings — counting, sorting, merging — with no floating point anywhere, so the
one-ULP libm divergence the *model* has to account for cannot arise here.

Ties in pair frequency are broken by the pair itself, sorted, so a dictionary's
iteration order can never decide a merge.

That is what lets the tokenizer be gated by **digest** while the model is gated
on metrics. See [reproducibility](reproducibility.md).

```python
tokenizer.digest()  # 'sha256:...' over the merges and the vocabulary
```

The digest is stored in the file and rechecked on load: a tokenizer edited by
hand — or truncated by a failed write — would otherwise produce a model that
trains happily against ids meaning something else.

## Choosing a vocabulary size

Measured on dev, with the neural model:

| Vocabulary | Naive Bayes | Neural |
| --- | --- | --- |
| **1024** | 0.907 | **0.918** |
| 2048 | 0.905 | 0.910 |
| 4096 | 0.910 | 0.901 |

Bigger is not better here. A larger vocabulary means rarer tokens, fewer
examples of each embedding, and more capacity to memorise. 1024 is the shipped
default.

`unfilled` in the training report says how many requested merges could not be
made because every remaining pair occurred once. Asking for 8,000 tokens and
getting 5,000 is a fact about the corpus worth knowing, so it is reported rather
than silently absorbed.

## See also

- [The model](model.md) — what consumes these tokens
- [Curation](curation.md) — the normalisation that happens first
- `python examples/tokenizer_demo.py` — the comparison, executing
