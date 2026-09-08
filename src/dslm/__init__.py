"""A small domain language model you can trust.

The model is real, small and trained from scratch. The reason to trust its
reported numbers is everything around it: a corpus with provenance on every
record, near-duplicates removed before splitting, and an evaluation that
**refuses to report a metric** when the split it was measured on overlaps the
data the model was trained on.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
