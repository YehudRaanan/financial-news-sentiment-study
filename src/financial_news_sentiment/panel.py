"""Build the fixed company-date return panel from sentiment and adjusted prices.

Price rows must already use the study's adjusted open and close values. The
module uses no network calls. This makes a frozen price snapshot auditable.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import add_earlier_news, aggregate_sentiment, prune_singletons
from .contracts import require_columns

TARGETS = {
    "prior7": (-7, -1),
    "future7": (1, 7),
    "future14": (1, 14),
    "future30": (1, 30),
}
GAP_START = pd.Timestamp("2025-06-15")
GAP_END = pd.Timestamp("2025-07-01")


@dataclass(frozen=True)
class PanelSettings:
    """Fixed settings from the final study."""

    market_peers: int = 100
    sector_peers: int = 5
    sentiment_market_peers: int = 20
    sentiment_sector_peers: int = 3
    history_sessions: int = 250
    history_gap_sessions: int = 30
    history_minimum: int = 100
    momentum_sessions: int = 21
    volatility_sessions: int = 60


def _window_sessions(
    date: pd.Timestamp, target: str, sessions: pd.DatetimeIndex
) -> pd.DatetimeIndex:
    """Return trading sessions inside the target calendar-day interval."""
    lower_days, upper_days = TARGETS[target]
    lower = date + pd.Timedelta(days=lower_days)
    upper = date + pd.Timedelta(days=upper_days)
    return sessions[(sessions >= lower) & (sessions <= upper)]


def _gap_overlap(start: pd.Timestamp, end: pd.Timestamp) -> bool:
    return start <= GAP_END and end >= GAP_START


def _peer_returns(prices: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Add leave-one-company-out market and sector daily returns."""
    frame = prices.merge(registry[["issuer", "sector"]], on="issuer", validate="many_to_one")
    frame = frame.sort_values(["issuer", "date"]).copy()
    frame["cc_return"] = frame.groupby("issuer")["close"].pct_change(fill_method=None)
    frame["oc_return"] = frame["close"] / frame["open"] - 1
    for return_name in ("cc_return", "oc_return"):
        valid = frame[return_name].notna().astype(int)
        values = frame[return_name].fillna(0)
        market_sum = values.groupby(frame["date"]).transform("sum") - values
        market_count = valid.groupby(frame["date"]).transform("sum") - valid
        sector_sum = values.groupby([frame["date"], frame["sector"]]).transform("sum") - values
        sector_count = valid.groupby([frame["date"], frame["sector"]]).transform("sum") - valid
        frame[f"market_{return_name}"] = market_sum / market_count.where(market_count.gt(0))
        frame[f"sector_{return_name}"] = sector_sum / sector_count.where(sector_count.gt(0))
        frame[f"market_{return_name}_peers"] = market_count
        frame[f"sector_{return_name}_peers"] = sector_count
    return frame


def _fit_exposures(
    history: pd.DataFrame, settings: PanelSettings
) -> tuple[float, float, float] | None:
    """Fit stock market exposure and residual sector exposure with intercepts."""
    columns = ["cc_return", "market_cc_return", "sector_cc_return"]
    history = history.dropna(subset=columns)
    history = history[
        history["market_cc_return_peers"].ge(settings.market_peers)
        & history["sector_cc_return_peers"].ge(settings.sector_peers)
    ].tail(settings.history_sessions)
    if len(history) < settings.history_minimum:
        return None
    market = history["market_cc_return"].to_numpy()
    sector = history["sector_cc_return"].to_numpy()
    stock = history["cc_return"].to_numpy()
    market_design = np.column_stack([np.ones(len(history)), market])
    sector_market_slope = np.linalg.lstsq(market_design, sector, rcond=None)[0][1]
    residual_sector = sector - sector_market_slope * market
    stock_design = np.column_stack([np.ones(len(history)), market, residual_sector])
    _, market_slope, sector_slope = np.linalg.lstsq(stock_design, stock, rcond=None)[0]
    return float(market_slope), float(sector_slope), float(sector_market_slope)


def _adjusted_return(
    company_prices: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    article_date: pd.Timestamp,
    target: str,
    settings: PanelSettings,
) -> float | None:
    """Calculate actual minus expected compounded return for one window."""
    window = _window_sessions(article_date, target, sessions)
    if window.empty:
        return None
    indexed = company_prices.set_index("date")
    if not window.isin(indexed.index).all():
        return None
    entry = window[0]
    entry_position = sessions.get_loc(entry)
    history_stop = entry_position - settings.history_gap_sessions + 1
    if history_stop <= 0:
        return None
    history_dates = sessions[max(0, history_stop - settings.history_sessions) : history_stop]
    exposure = _fit_exposures(indexed.reindex(history_dates), settings)
    if exposure is None:
        return None
    market_slope, sector_slope, sector_market_slope = exposure
    observed = indexed.reindex(window)
    peer_ok = observed["market_cc_return_peers"].ge(settings.market_peers)
    peer_ok &= observed["sector_cc_return_peers"].ge(settings.sector_peers)
    if not peer_ok.all():
        return None
    market = observed["market_cc_return"].to_numpy(copy=True)
    sector = observed["sector_cc_return"].to_numpy(copy=True)
    market[0] = observed["market_oc_return"].iloc[0]
    sector[0] = observed["sector_oc_return"].iloc[0]
    expected_daily = market_slope * market + sector_slope * (sector - sector_market_slope * market)
    if not np.isfinite(expected_daily).all() or (expected_daily <= -1).any():
        return None
    actual = observed["close"].iloc[-1] / observed["open"].iloc[0] - 1
    expected = np.prod(1 + expected_daily) - 1
    return float(actual - expected)


def build_panel(
    pairs: pd.DataFrame,
    prices: pd.DataFrame,
    registry: pd.DataFrame,
    settings: PanelSettings | None = None,
) -> pd.DataFrame:
    """Build the final eligible panel from frozen sentiment and price inputs."""
    settings = settings or PanelSettings()
    require_columns(prices, ["issuer", "date", "open", "close"], "prices")
    require_columns(registry, ["issuer", "sector"], "registry")
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices = _peer_returns(prices, registry)
    sessions = pd.DatetimeIndex(sorted(prices["date"].unique()))
    daily = add_earlier_news(aggregate_sentiment(pairs))
    daily = daily.merge(registry[["issuer", "sector"]], on="issuer", validate="many_to_one")
    directional = daily["directional"].gt(0)
    market_count = directional.groupby(daily["date"]).transform("sum") - directional.astype(int)
    sector_count = directional.groupby([daily["date"], daily["sector"]]).transform(
        "sum"
    ) - directional.astype(int)
    daily["sentiment_peer_support"] = market_count.ge(settings.sentiment_market_peers)
    daily["sentiment_peer_support"] &= sector_count.ge(settings.sentiment_sector_peers)

    price_groups = {issuer: frame for issuer, frame in prices.groupby("issuer")}
    rows = []
    for row in daily.itertuples(index=False):
        if row.directional == 0 or not row.sentiment_peer_support or pd.isna(row.log_past_news):
            continue
        date = pd.Timestamp(row.date)
        if GAP_START <= date <= GAP_END:
            continue
        if _gap_overlap(date - pd.Timedelta(days=14), date - pd.Timedelta(days=8)):
            continue
        if _gap_overlap(date + pd.Timedelta(days=1), date + pd.Timedelta(days=30)):
            continue
        company_prices = price_groups.get(row.issuer)
        if company_prices is None:
            continue
        outcomes = {
            target: _adjusted_return(company_prices, sessions, date, target, settings)
            for target in TARGETS
        }
        if any(value is None for value in outcomes.values()):
            continue
        prior = _window_sessions(date, "prior7", sessions)
        prior_entry = sessions.get_loc(prior[0])
        indexed = company_prices.set_index("date").reindex(sessions)
        prior_returns = indexed["cc_return"].iloc[:prior_entry]
        momentum_values = prior_returns.tail(settings.momentum_sessions)
        volatility_values = prior_returns.tail(settings.volatility_sessions)
        if momentum_values.isna().any() or volatility_values.isna().any():
            continue
        result = row._asdict()
        result.update(
            momentum=float(np.prod(1 + momentum_values) - 1),
            volatility=float(volatility_values.std(ddof=1)),
            **{f"adjusted_{target}": value for target, value in outcomes.items()},
        )
        rows.append(result)
    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel
    return prune_singletons(panel).sort_values(["issuer", "date"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panel = build_panel(
        pd.read_parquet(args.pairs),
        pd.read_parquet(args.prices),
        pd.read_csv(args.registry),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
