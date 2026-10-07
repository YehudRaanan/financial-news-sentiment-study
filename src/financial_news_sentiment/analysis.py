"""Build daily sentiment measures and estimate the report regressions.

The estimator absorbs company and recorded-date effects. It uses the same
company and fixed 28-day calendar-block covariance calculation as the final
report. Inputs are explicit so the module does not depend on local paths.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from .contracts import LABELS, require_columns, validate_probabilities

BASE_CONTROLS = ("momentum", "volatility", "log_past_news")
FOCAL_TERMS = ("positive_share", "negative_share")


def aggregate_sentiment(pairs: pd.DataFrame) -> pd.DataFrame:
    """Convert article-company labels to one row per company and date.

    NONE assessments remain in the volume count but do not enter the share
    denominator. This matches the study definition.
    """
    require_columns(pairs, ["issuer", "date", "label"], "sentiment pairs")
    pairs = pairs.copy()
    pairs["date"] = pd.to_datetime(pairs["date"]).dt.normalize()
    counts = (
        pairs.groupby(["issuer", "date", "label"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=LABELS, fill_value=0)
        .reset_index()
    )
    counts["pairs"] = counts[list(LABELS)].sum(axis=1)
    counts["directional"] = counts[["positive", "negative", "mix"]].sum(axis=1)
    for label in ("positive", "negative", "mix"):
        counts[f"{label}_share"] = counts[label] / counts["directional"].where(
            counts["directional"].gt(0)
        )
    return counts


def add_earlier_news(daily: pd.DataFrame) -> pd.DataFrame:
    """Count all saved labels from 14 through 8 calendar days before each date."""
    require_columns(daily, ["issuer", "date", "pairs"], "daily sentiment")
    daily = daily.copy().sort_values(["issuer", "date"])
    dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    issuers = sorted(daily["issuer"].unique())
    grid = pd.MultiIndex.from_product([issuers, dates], names=["issuer", "date"])
    volume = daily.set_index(["issuer", "date"])["pairs"].reindex(grid, fill_value=0)
    earlier = volume.groupby(level=0).transform(
        lambda values: values.rolling(7, min_periods=7).sum().shift(8)
    )
    result = daily.set_index(["issuer", "date"])
    result["earlier_news"] = earlier.reindex(result.index)
    result["log_past_news"] = np.log1p(result["earlier_news"])
    return result.reset_index()


def prune_singletons(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove company or date groups with one row until the sample is stable."""
    frame = frame.copy()
    while len(frame):
        keep = frame.groupby("issuer")["issuer"].transform("size").gt(1)
        keep &= frame.groupby("date")["date"].transform("size").gt(1)
        if keep.all():
            break
        frame = frame.loc[keep].copy()
    return frame


def _within(values: np.ndarray, frame: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    """Absorb company and date effects by alternating weighted demeaning."""
    result = values.copy()
    groups = [pd.factorize(frame["issuer"])[0], pd.factorize(frame["date"])[0]]
    for _ in range(1_000):
        previous = result.copy()
        for group in groups:
            denominator = np.bincount(group, weights=weights)
            for column in range(result.shape[1]):
                means = np.bincount(group, weights=weights * result[:, column])
                result[:, column] -= means[group] / denominator[group]
        if np.max(np.abs(result - previous)) < 1e-10:
            return result
    raise RuntimeError("Fixed-effect absorption did not converge")


def _cluster_variance(
    influence: np.ndarray, frame: pd.DataFrame, finite_sample_adjustment: float
) -> float:
    """Return company plus calendar-block covariance minus their intersection."""

    def sum_squares(groups: pd.Series) -> float:
        codes = pd.factorize(groups)[0]
        count = len(np.unique(codes))
        if count <= 1:
            return math.nan
        totals = np.bincount(codes, weights=influence)
        return float(totals @ totals) * count / (count - 1)

    company = sum_squares(frame["issuer"])
    block = sum_squares(frame["calendar_block"])
    intersection = sum_squares(
        frame["issuer"].astype(str) + "|" + frame["calendar_block"].astype(str)
    )
    return finite_sample_adjustment * (company + block - intersection)


def fit_period(frame: pd.DataFrame, outcome: str) -> list[dict[str, float | int | str]]:
    """Fit the joint positive and negative share model for one sample."""
    names = [*FOCAL_TERMS, *BASE_CONTROLS]
    require_columns(frame, ["issuer", "date", outcome, *names], "analysis panel")
    frame = prune_singletons(frame.dropna(subset=[outcome, *names]).copy())
    frame["calendar_block"] = ((frame["date"] - pd.Timestamp("2023-01-01")).dt.days // 28).astype(
        int
    )
    weights = np.ones(len(frame))
    transformed = _within(frame[[outcome, *names]].to_numpy(float), frame, weights)
    y, x = transformed[:, 0], transformed[:, 1:]
    scale = np.sqrt(np.average(x * x, axis=0, weights=weights))
    keep: list[int] = []
    for index in range(len(names)):
        candidates = keep + [index]
        scaled = x[:, candidates] / scale[candidates]
        if scale[index] > 1e-10 and np.linalg.matrix_rank(scaled, tol=1e-8) > len(keep):
            keep.append(index)
    if not set(range(len(FOCAL_TERMS))).issubset(keep):
        raise ValueError("A focal sentiment term is not identified")
    x = x[:, keep]
    kept_names = [names[index] for index in keep]
    bread = np.linalg.inv(x.T @ x)
    beta = bread @ (x.T @ y)
    residual = y - x @ beta
    influence = (x * residual[:, None]) @ bread
    fixed_effect_rank = frame["issuer"].nunique() + frame["date"].nunique() - 1
    adjustment = (len(frame) - 1) / max(1, len(frame) - fixed_effect_rank - len(kept_names))
    degrees = min(frame["issuer"].nunique(), frame["calendar_block"].nunique()) - 1
    critical = t.ppf(0.975, degrees)
    rows = []
    for term in FOCAL_TERMS:
        index = kept_names.index(term)
        variance = _cluster_variance(influence[:, index], frame, adjustment)
        standard_error = math.sqrt(variance) if variance > 0 else math.nan
        estimate = float(beta[index])
        rows.append(
            {
                "term": term,
                "coefficient": estimate,
                "se": standard_error,
                "low": estimate - critical * standard_error,
                "high": estimate + critical * standard_error,
                "n": len(frame),
                "issuers": frame["issuer"].nunique(),
                "dates": frame["date"].nunique(),
                "blocks": frame["calendar_block"].nunique(),
                "df": degrees,
            }
        )
    return rows


def estimate_periods(panel: pd.DataFrame) -> pd.DataFrame:
    """Estimate before and after models for full, earlier, and 2026 samples."""
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    periods = {
        "full": panel,
        "earlier": panel.loc[panel["date"].lt("2026-01-01")],
        "recent": panel.loc[panel["date"].ge("2026-01-01")],
    }
    rows = []
    for period, subset in periods.items():
        for target, outcome in (
            ("prior7", "adjusted_prior7"),
            ("future7", "adjusted_future7"),
        ):
            for row in fit_period(subset, outcome):
                rows.append({"period": period, "target": target, **row})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, help="Optional sentiment-pair Parquet file")
    parser.add_argument("--panel", type=Path, required=True, help="Prepared return panel")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pairs:
        pairs = pd.read_parquet(args.pairs)
        validate_probabilities(pairs)
    panel = pd.read_parquet(args.panel)
    result = estimate_periods(panel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
