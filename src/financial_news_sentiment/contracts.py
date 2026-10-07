"""Shared data contracts and validation helpers.

The public repository does not contain the research corpus. These checks fail
early when a user supplies a file with missing or inconsistent fields.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ("positive", "negative", "mix", "none")
PROBABILITY_COLUMNS = LABELS


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    """Raise a clear error when an input does not satisfy its contract."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(missing)}")


def validate_probabilities(frame: pd.DataFrame) -> None:
    """Validate the four saved Jev probabilities and selected category."""
    require_columns(frame, ["label", *PROBABILITY_COLUMNS], "sentiment pairs")
    values = frame.loc[:, PROBABILITY_COLUMNS]
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Probabilities must be finite")
    if not values.ge(0).all().all() or not values.le(1).all().all():
        raise ValueError("Probabilities must be between zero and one")
    if not np.isclose(values.sum(axis=1), 1, atol=0.021).all():
        raise ValueError("Probabilities must sum to one within rounding tolerance")
    if not frame["label"].isin(LABELS).all():
        raise ValueError("Unknown sentiment category")
    selected = values.to_numpy().argmax(axis=1)
    expected = np.asarray(LABELS, dtype=object)[selected]
    if not np.array_equal(frame["label"].to_numpy(), expected):
        raise ValueError("Selected category is not a maximum saved probability")


def sha256(path: Path) -> str:
    """Return a file checksum without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
