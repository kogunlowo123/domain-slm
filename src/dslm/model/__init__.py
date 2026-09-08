"""The models: a neural classifier and the linear control it has to beat."""

from __future__ import annotations

from dslm.model.features import Encoded, LabelSpace, encode, label_space
from dslm.model.linear import NaiveBayes
from dslm.model.neural import Classifier, TrainingHistory, accuracy

__all__ = [
    "Classifier",
    "Encoded",
    "LabelSpace",
    "NaiveBayes",
    "TrainingHistory",
    "accuracy",
    "encode",
    "label_space",
]
