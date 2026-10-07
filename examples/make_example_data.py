"""Create the small synthetic example that is safe to publish."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from financial_news_sentiment.analysis import estimate_periods


def build(output: Path) -> None:
    """Write deterministic inputs with the same columns as the full study."""
    rng = np.random.default_rng(20261007)
    output.mkdir(parents=True, exist_ok=True)
    issuers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    dates = pd.to_datetime(
        [
            "2023-06-16",
            "2023-09-15",
            "2024-01-12",
            "2024-05-17",
            "2024-09-20",
            "2025-01-17",
            "2025-05-16",
            "2025-09-19",
            "2026-01-16",
            "2026-02-20",
            "2026-03-20",
            "2026-04-17",
            "2026-05-15",
            "2026-06-05",
            "2026-06-12",
            "2026-06-18",
        ]
    )
    domains = ["example-a.test", "example-b.test", "example-c.test", "example-d.test"]
    labels = ["positive", "negative", "mix", "none"]
    probability_rows = {
        "positive": (0.72, 0.10, 0.10, 0.08),
        "negative": (0.10, 0.72, 0.10, 0.08),
        "mix": (0.18, 0.17, 0.57, 0.08),
        "none": (0.08, 0.08, 0.09, 0.75),
    }
    articles = []
    pairs = []
    panel = []
    recap = []
    for date_index, date in enumerate(dates):
        for issuer_index, issuer in enumerate(issuers):
            article_id = f"example-{date_index:02d}-{issuer.lower()}"
            label = labels[(date_index + 2 * issuer_index) % len(labels)]
            probabilities = np.asarray(probability_rows[label], dtype=float)
            noise = rng.normal(0, 0.006, size=4)
            probabilities = np.clip(probabilities + noise, 0.001, None)
            probabilities /= probabilities.sum()
            selected = labels[int(probabilities.argmax())]
            pair_id = f"{article_id}|{issuer}"
            articles.append(
                {
                    "article_id": article_id,
                    "date": date,
                    "domain": domains[(date_index + issuer_index) % len(domains)],
                    "words": 80 + ((date_index * 227 + issuer_index * 149) % 8_000),
                    "company_count": 1 + ((date_index + issuer_index) % 5),
                }
            )
            pairs.append(
                {
                    "pair_id": pair_id,
                    "article_id": article_id,
                    "issuer": issuer,
                    "date": date,
                    "domain": domains[(date_index + issuer_index) % len(domains)],
                    "label": selected,
                    **dict(zip(labels, probabilities, strict=True)),
                }
            )
            recap.append(
                {
                    "pair_id": pair_id,
                    "category": "recap" if (date_index + issuer_index) % 9 == 0 else "retained",
                }
            )
            positive_share = ((date_index + 3 * issuer_index) % 11) / 10
            negative_share = ((2 * date_index + issuer_index + 1) % 9) / 10
            mix_share = max(0.0, 1 - min(1.0, positive_share + negative_share))
            momentum = rng.normal(0, 0.08)
            volatility = 0.01 + rng.random() * 0.04
            earlier_news = np.log1p((date_index + issuer_index) % 8)
            company_effect = (issuer_index - 2.5) * 0.0004
            date_effect = (date_index - 7.5) * 0.0001
            before = 0.006 * positive_share - 0.007 * negative_share + company_effect + date_effect
            after = 0.0003 * positive_share + 0.0004 * negative_share + company_effect + date_effect
            panel.append(
                {
                    "issuer": issuer,
                    "date": date,
                    "positive_share": positive_share,
                    "negative_share": negative_share,
                    "mix_share": mix_share,
                    "momentum": momentum,
                    "volatility": volatility,
                    "log_past_news": earlier_news,
                    "adjusted_prior7": before + rng.normal(0, 0.002),
                    "adjusted_future7": after + rng.normal(0, 0.002),
                }
            )
    csv_options = {"index": False, "lineterminator": "\n"}
    pd.DataFrame(articles).to_csv(output / "articles.csv", date_format="%Y-%m-%d", **csv_options)
    pd.DataFrame(pairs).to_csv(
        output / "sentiment_pairs.csv", date_format="%Y-%m-%d", **csv_options
    )
    panel_frame = pd.DataFrame(panel)
    panel_frame.to_csv(output / "analysis_panel.csv", date_format="%Y-%m-%d", **csv_options)
    pd.DataFrame(recap).to_csv(output / "recap_screen.csv", **csv_options)
    pd.DataFrame(
        [{"measure": "precision", "value": 0.90}, {"measure": "recall", "value": 0.86}]
    ).to_csv(output / "identity_validation.csv", **csv_options)
    estimate_periods(panel_frame).to_csv(output / "period_coefficients.csv", **csv_options)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
