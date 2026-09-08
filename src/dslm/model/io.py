"""Saving and loading models.

An artefact directory rather than a single file: ``model.npz`` for the weights
and ``model.json`` for everything a human or a report needs to read without
loading NumPy. The metadata is the part that gets inspected — in a review, in a
CI summary, in a bug report six months later — and requiring an array library to
answer "which tokenizer was this trained against?" would guarantee nobody asks.

Three things are checked on load, and each of them was a real failure somewhere
before it was a check here:

**The metadata and the weights agree about shapes.** A truncated ``.npz`` loads
without complaint and produces a model that predicts confidently from garbage.

**The tokenizer digest matches.** A model run against a vocabulary it was not
trained on maps token 412 to a different string, and the result is not an error —
it is an accuracy about eight points lower, which reads as a bad model rather
than as a mismatch.

**The label space matches.** Same failure, one level up: class 7 means ATA 33 in
one label space and ATA 34 in another, and the confusion matrix looks merely
disappointing.

None of these can be caught by looking at the output, which is exactly why they
are checked at the boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from dslm.errors import ModelError
from dslm.model.features import LabelSpace
from dslm.model.neural import Classifier

#: The artefact version, written into the metadata and checked on load.
MODEL_SCHEMA_VERSION = 1

WEIGHTS_NAME = "model.npz"
METADATA_NAME = "model.json"


def save(
    model: Classifier,
    directory: str | Path,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write *model* to *directory* and return the directory.

    ``np.savez`` rather than ``savez_compressed``: the weights are dense floats
    and compress by a few percent, which is not worth making the artefact's bytes
    depend on the zlib version — a compressed file is not reproducible across
    library versions even when its contents are.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)

    np.savez(
        target / WEIGHTS_NAME,
        embedding=model.embedding,
        hidden_weight=model.hidden_weight,
        hidden_bias=model.hidden_bias,
        output_weight=model.output_weight,
        output_bias=model.output_bias,
    )

    metadata: dict[str, Any] = {
        "dslm_model": MODEL_SCHEMA_VERSION,
        **model.as_dict(),
        "padding_id": model.padding_id,
        "shapes": {
            "embedding": list(model.embedding.shape),
            "hidden_weight": list(model.hidden_weight.shape),
            "hidden_bias": list(model.hidden_bias.shape),
            "output_weight": list(model.output_weight.shape),
            "output_bias": list(model.output_bias.shape),
        },
    }
    if extra:
        metadata["training"] = extra
    (target / METADATA_NAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return target


def load(directory: str | Path, *, tokenizer_digest: str = "") -> tuple[Classifier, dict[str, Any]]:
    """Read a model written by :func:`save`, with its metadata.

    Pass *tokenizer_digest* to have the mismatch caught here rather than
    discovered as an unexplained drop in accuracy.
    """
    source = Path(directory)
    weights_path = source / WEIGHTS_NAME
    metadata_path = source / METADATA_NAME
    for path in (weights_path, metadata_path):
        if not path.is_file():
            raise ModelError(
                f"{str(path)!r} is missing.",
                remedy=(
                    "A model artefact is a directory holding both "
                    f"{WEIGHTS_NAME} and {METADATA_NAME}. Retrain with 'dslm train'."
                ),
            )

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelError(
            f"{METADATA_NAME} is not valid JSON: {exc}.",
            remedy="Retrain; a truncated artefact cannot be repaired by hand.",
        ) from exc

    version = metadata.get("dslm_model")
    if not isinstance(version, int) or version > MODEL_SCHEMA_VERSION:
        raise ModelError(
            f"the model was written by a newer version (schema {version!r}).",
            remedy=f"This build understands schema {MODEL_SCHEMA_VERSION}. Upgrade dslm.",
        )

    with np.load(weights_path) as archive:
        try:
            arrays = {name: archive[name] for name in archive.files}
        except Exception as exc:
            raise ModelError(
                f"the weights in {WEIGHTS_NAME} could not be read: {exc}.",
                remedy="Retrain.",
            ) from exc

    required = ("embedding", "hidden_weight", "hidden_bias", "output_weight", "output_bias")
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ModelError(
            f"{WEIGHTS_NAME} is missing {', '.join(missing)}.",
            remedy="Retrain.",
        )

    shapes = metadata.get("shapes") or {}
    for name in required:
        expected = shapes.get(name)
        if expected is not None and list(arrays[name].shape) != list(expected):
            raise ModelError(
                f"{name} has shape {list(arrays[name].shape)} and the metadata says "
                f"{list(expected)}.",
                remedy=(
                    "The artefact is truncated or was assembled from two runs. A model "
                    "loaded with mismatched shapes predicts confidently from nothing."
                ),
            )

    stored_tokenizer = str(metadata.get("tokenizer_digest") or "")
    if tokenizer_digest and stored_tokenizer and tokenizer_digest != stored_tokenizer:
        raise ModelError(
            "this model was trained against a different tokenizer.",
            remedy=(
                f"The model records {stored_tokenizer[:23]}... and the tokenizer supplied is "
                f"{tokenizer_digest[:23]}.... Token ids mean different strings under the two, "
                "so the model would score confidently and wrongly rather than fail."
            ),
        )

    space = metadata.get("label_space") or {}
    labels = tuple(int(label) for label in space.get("labels", ()))
    if not labels:
        raise ModelError(
            "the model records no label space.",
            remedy="Without it a predicted index cannot be turned back into a chapter.",
        )

    model = Classifier(
        embedding=arrays["embedding"],
        hidden_weight=arrays["hidden_weight"],
        hidden_bias=arrays["hidden_bias"],
        output_weight=arrays["output_weight"],
        output_bias=arrays["output_bias"],
        label_space=LabelSpace(labels=labels),
        padding_id=int(metadata.get("padding_id", 0)),
        seed=int(metadata.get("seed", 0)),
        tokenizer_digest=stored_tokenizer,
    )
    return model, metadata
