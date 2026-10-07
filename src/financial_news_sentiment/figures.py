"""Create every figure used in the final report and probability annex."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from .contracts import PROBABILITY_COLUMNS, require_columns, sha256, validate_probabilities

COLORS = {
    "positive": "#2b7a66",
    "negative": "#b55252",
    "mix": "#8064a2",
    "none": "#6b7280",
}
PERIOD_COLORS = {"Full": "#245d78", "2023-25": "#5d7f4e", "2026": "#d27a35"}
DISPLAY_LABELS = ["POSITIVE", "NEGATIVE", "MIX", "NONE"]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 120,
    }
)


def _clean(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d8dee3", linewidth=0.6, alpha=0.8)


def _save(fig: plt.Figure, output: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(
        output / f"{name}.png",
        dpi=220,
        bbox_inches="tight",
        metadata={"Software": "financial-news-sentiment-study"},
    )
    plt.close(fig)


def article_lengths(articles: pd.DataFrame, output: Path) -> None:
    require_columns(articles, ["words"], "articles")
    words = articles["words"].to_numpy()
    q1, median, q3 = np.quantile(words, [0.25, 0.5, 0.75])
    fig, ax = plt.subplots(figsize=(9, 3.7))
    bins = np.geomspace(max(1, words.min() * 0.9), words.max() * 1.02, 55)
    ax.hist(
        words, bins=bins, color="#27627b", alpha=0.85, weights=np.full(len(words), 100 / len(words))
    )
    ax.set_xscale("log")
    ax.set_xlim(bins[0], bins[-1])
    for value, label, color, y in [
        (q1, f"Lower quarter: {q1:,.0f}", "#bc6e34", 0.96),
        (median, f"Median: {median:,.0f}", "#213f55", 0.84),
        (q3, f"Upper quarter: {q3:,.0f}", "#7d515f", 0.72),
    ]:
        ax.axvline(value, color=color, linewidth=1.5, linestyle="--")
        ax.text(0.98, y, label, transform=ax.transAxes, ha="right", va="top", color=color)
    ax.set_xlabel("Words per saved article (logarithmic scale)")
    ax.set_ylabel("Articles in each length interval (%)")
    ax.set_title(f"Article lengths | {len(words):,} saved articles")
    _clean(ax)
    _save(fig, output, "article_lengths_integrated")


def period_coefficients(coefficients: pd.DataFrame, output: Path) -> None:
    require_columns(
        coefficients, ["target", "period", "term", "coefficient", "low", "high"], "coefficients"
    )
    index = coefficients.set_index(["target", "period", "term"])
    periods = [
        ("full", "Full\nhistory"),
        ("earlier", "Jun 2023\n– Dec 2025"),
        ("recent", "2026\nJan–Jun"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.05))
    for ax, target, title in zip(
        axes,
        ["prior7", "future7"],
        ["Before recorded article date", "After recorded article date"],
        strict=True,
    ):
        for term, color, shift, name in [
            ("positive_share", "#286486", -0.09, "Positive share"),
            ("negative_share", "#bd6330", 0.09, "Negative share"),
        ]:
            for position, (period, _) in enumerate(periods):
                row = index.loc[(target, period, term)]
                value, low, high = [100 * float(row[key]) for key in ("coefficient", "low", "high")]
                ax.errorbar(
                    position + shift,
                    value,
                    yerr=[[value - low], [high - value]],
                    fmt="o",
                    color=color,
                    capsize=4,
                    markersize=6,
                    label=name if position == 0 else None,
                )
        ax.axhline(0, color="#74828a", linewidth=0.8)
        ax.set_xticks(range(3), [label for _, label in periods])
        ax.set_title(title)
        ax.set_ylabel("Return difference (percentage points)")
        _clean(ax)
    axes[1].legend(frameon=False)
    _save(fig, output, "period_slopes")


def corpus_overview(articles: pd.DataFrame, pairs: pd.DataFrame, output: Path) -> None:
    require_columns(articles, ["date", "domain"], "articles")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
    annual = articles.assign(year=pd.to_datetime(articles["date"]).dt.year).groupby("year").size()
    axes[0].bar(annual.index.astype(str), annual.values, color="#286486")
    axes[0].set_title("Saved news volume by year")
    axes[0].set_ylabel("Articles")
    counts = pairs["label"].value_counts().reindex(PROBABILITY_COLUMNS, fill_value=0)
    axes[1].bar(DISPLAY_LABELS, counts.values, color=[COLORS[x] for x in PROBABILITY_COLUMNS])
    axes[1].set_title("Jev company-specific labels")
    axes[1].set_ylabel("Assessments")
    for ax in axes:
        _clean(ax)
    _save(fig, output, "corpus_jev")

    sources = articles["domain"].value_counts().head(12).sort_values()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(sources.index, sources.values, color="#286486")
    ax.set_xlabel("Saved articles")
    ax.set_title("Largest source domains in the saved collection")
    _clean(ax)
    _save(fig, output, "corpus_sources")


def validation_figure(validation: pd.DataFrame, output: Path) -> None:
    require_columns(validation, ["measure", "value"], "validation")
    values = validation.set_index("measure")["value"].reindex(["precision", "recall"])
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    bars = ax.bar(["Precision", "Recall"], values, color=["#286486", "#bd6330"])
    ax.bar_label(bars, labels=[f"{100 * value:.1f}%" for value in values], padding=3)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("Company linking check")
    _clean(ax)
    _save(fig, output, "jev_labeled50_precision_recall")


def probability_figures(
    pairs: pd.DataFrame,
    articles: pd.DataFrame,
    panel: pd.DataFrame,
    recap: pd.DataFrame,
    output: Path,
) -> None:
    validate_probabilities(pairs)
    pairs = pairs.copy()
    pairs["date"] = pd.to_datetime(pairs["date"])
    pairs["period"] = np.where(pairs["date"].dt.year.eq(2026), "2026", "2023-25")
    pairs["top_probability"] = pairs[list(PROBABILITY_COLUMNS)].max(axis=1)
    ordered = np.sort(pairs[list(PROBABILITY_COLUMNS)].to_numpy(), axis=1)
    pairs["top_margin"] = ordered[:, -1] - ordered[:, -2]
    pairs["probability_balance"] = pairs["positive"] - pairs["negative"]

    means = pd.concat(
        {
            "Full": pairs[list(PROBABILITY_COLUMNS)].mean(),
            "2026": pairs.loc[pairs["period"].eq("2026"), list(PROBABILITY_COLUMNS)].mean(),
        },
        axis=1,
    ).T
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(4)
    width = 0.34
    for index, period in enumerate(["Full", "2026"]):
        values = means.loc[period].to_numpy()
        bars = ax.bar(
            x + (index - 0.5) * width, values, width, label=period, color=PERIOD_COLORS[period]
        )
        ax.bar_label(bars, labels=[f"{100 * value:.1f}%" for value in values], padding=3)
    ax.set_xticks(x, DISPLAY_LABELS)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Mean saved probability")
    ax.set_title("Mean category probabilities | full corpus and 2026")
    ax.legend(frameon=False)
    _clean(ax)
    _save(fig, output, "prob_01_mean_profile")

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.6), sharex=True, sharey=True)
    bins = np.linspace(0.25, 1, 31)
    for ax, label, title in zip(axes.flat, PROBABILITY_COLUMNS, DISPLAY_LABELS, strict=True):
        for period in ("Full", "2026"):
            mask = pairs["label"].eq(label)
            if period == "2026":
                mask &= pairs["period"].eq("2026")
            values = pairs.loc[mask, "top_probability"]
            if len(values):
                ax.hist(
                    values,
                    bins=bins,
                    weights=np.full(len(values), 100 / len(values)),
                    histtype="step",
                    linewidth=1.8,
                    color=PERIOD_COLORS[period],
                    label=f"{period} n={len(values):,}",
                )
        ax.set_title(title)
        ax.legend(frameon=False)
        _clean(ax)
    axes[1, 0].set_xlabel("Highest of the four saved probabilities")
    axes[1, 1].set_xlabel("Highest of the four saved probabilities")
    axes[0, 0].set_ylabel("Assessments in bin (%)")
    axes[1, 0].set_ylabel("Assessments in bin (%)")
    fig.suptitle("Highest saved probability by selected category", y=1.01)
    _save(fig, output, "prob_02_top_distribution")

    fig, ax = plt.subplots(figsize=(9, 3.8))
    balance_bins = np.linspace(-1, 1, 51)
    for period, mask in [
        ("Full", np.ones(len(pairs), dtype=bool)),
        ("2023-25", pairs["period"].eq("2023-25")),
        ("2026", pairs["period"].eq("2026")),
    ]:
        values = pairs.loc[mask, "probability_balance"]
        counts, edges = np.histogram(values, bins=balance_bins)
        ax.step(
            (edges[:-1] + edges[1:]) / 2,
            100 * counts / counts.sum(),
            where="mid",
            linewidth=1.8,
            color=PERIOD_COLORS[period],
            label=period,
        )
    ax.axvline(0, color="#334155", linewidth=0.8)
    ax.set_xlabel("POSITIVE probability minus NEGATIVE probability")
    ax.set_ylabel("Assessments in bin (%)")
    ax.set_title("Probability balance distribution")
    ax.legend(frameon=False)
    _clean(ax)
    _save(fig, output, "prob_03_balance_distribution")

    fig, ax = plt.subplots(figsize=(9, 3.8))
    for period, mask in [
        ("Full", np.ones(len(pairs), dtype=bool)),
        ("2026", pairs["period"].eq("2026")),
    ]:
        values = pairs.loc[mask, "top_margin"]
        ax.hist(
            values,
            bins=np.linspace(0, 1, 41),
            weights=np.full(len(values), 100 / len(values)),
            histtype="step",
            linewidth=2,
            color=PERIOD_COLORS[period],
            label=period,
        )
    ax.set_xlabel("Highest probability minus second-highest probability")
    ax.set_ylabel("Assessments in bin (%)")
    ax.set_title("Separation between the first and second category")
    ax.legend(frameon=False)
    _clean(ax)
    _save(fig, output, "prob_04_margin_distribution")

    monthly = (
        pairs.set_index("date").groupby(pd.Grouper(freq="MS"))[list(PROBABILITY_COLUMNS)].mean()
    )
    monthly["balance"] = monthly["positive"] - monthly["negative"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.3), sharex=True)
    for probability in PROBABILITY_COLUMNS:
        axes[0].plot(
            monthly.index,
            monthly[probability],
            color=COLORS[probability],
            label=probability.upper(),
            linewidth=1.7,
        )
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_title("Monthly category probabilities")
    axes[0].legend(frameon=False, ncol=4)
    axes[1].plot(monthly.index, monthly["balance"], color="#245d78", linewidth=1.8)
    axes[1].axhline(0, color="#334155", linewidth=0.8)
    axes[1].set_title("Monthly probability balance")
    axes[1].set_xlabel("Recorded article month")
    for ax in axes:
        _clean(ax)
    _save(fig, output, "prob_05_monthly_trends")

    top_domains = pairs["domain"].value_counts().head(10).index
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.5))
    for ax, period, frame in [
        (axes[0], "Full", pairs),
        (axes[1], "2026", pairs[pairs["period"].eq("2026")]),
    ]:
        table = (
            frame[frame["domain"].isin(top_domains)]
            .groupby("domain")[list(PROBABILITY_COLUMNS)]
            .mean()
            .reindex(top_domains)
        )
        image = ax.imshow(table.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=0.75)
        ax.set_yticks(range(len(table)), table.index)
        ax.set_xticks(range(4), DISPLAY_LABELS)
        ax.set_title(period)
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, label="Mean saved probability")
    fig.suptitle("Mean category probabilities by source domain")
    fig.subplots_adjust(left=0.22, right=0.92, top=0.93, bottom=0.08, hspace=0.3)
    fig.savefig(
        output / "prob_06_source_heatmap.png",
        dpi=220,
        bbox_inches="tight",
        metadata={"Software": "financial-news-sentiment-study"},
    )
    plt.close(fig)

    length_pairs = pairs.merge(
        articles[["article_id", "words"]], on="article_id", validate="many_to_one"
    )
    length_pairs["length_decile"] = pd.qcut(
        length_pairs["words"], 10, labels=False, duplicates="drop"
    )
    length_stats = length_pairs.groupby("length_decile").agg(
        median_words=("words", "median"),
        highest_probability=("top_probability", "mean"),
        top_margin=("top_margin", "mean"),
        none_probability=("none", "mean"),
    )
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for column, label, color in [
        ("highest_probability", "Highest probability", "#245d78"),
        ("top_margin", "Top-minus-second margin", "#d27a35"),
        ("none_probability", "NONE probability", COLORS["none"]),
    ]:
        ax.plot(
            length_stats["median_words"],
            length_stats[column],
            marker="o",
            linewidth=1.8,
            label=label,
            color=color,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Median words in each assessment-weighted length decile (log scale)")
    ax.set_ylabel("Mean probability or margin")
    ax.set_title("Saved article length and probability summaries")
    ax.legend(frameon=False)
    _clean(ax)
    _save(fig, output, "prob_07_length_relation")

    company_pairs = pairs.merge(
        articles[["article_id", "company_count"]], on="article_id", validate="many_to_one"
    )
    company_pairs["company_group"] = (
        company_pairs["company_count"].clip(upper=5).map({1: "1", 2: "2", 3: "3", 4: "4", 5: "5+"})
    )
    company_stats = (
        company_pairs.groupby("company_group", observed=True)
        .agg(highest_probability=("top_probability", "mean"), top_margin=("top_margin", "mean"))
        .reindex(["1", "2", "3", "4", "5+"])
    )
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(company_stats))
    ax.plot(
        x,
        company_stats["highest_probability"],
        marker="o",
        label="Highest probability",
        color="#245d78",
    )
    ax.plot(
        x, company_stats["top_margin"], marker="o", label="Top-minus-second margin", color="#d27a35"
    )
    ax.set_xticks(x, company_stats.index)
    ax.set_xlabel("Registry companies linked to the article")
    ax.set_ylabel("Mean probability or margin")
    ax.set_title("Probability summaries by number of linked companies")
    ax.legend(frameon=False)
    _clean(ax)
    _save(fig, output, "prob_08_company_count_relation")

    require_columns(recap, ["pair_id", "category"], "recap screen")
    eligible = pairs.merge(recap[["pair_id", "category"]], on="pair_id", validate="one_to_one")
    eligible["screen_group"] = np.where(
        eligible["category"].eq("recap"), "Recap marked", "Retained"
    )
    recap_stats = (
        eligible.groupby("screen_group")
        .agg(
            assessments=("pair_id", "size"),
            positive=("positive", "mean"),
            negative=("negative", "mean"),
            mix=("mix", "mean"),
            none=("none", "mean"),
            balance=("probability_balance", "mean"),
        )
        .reindex(["Retained", "Recap marked"])
    )
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.1), gridspec_kw={"width_ratios": [2.2, 1]})
    x = np.arange(2)
    width = 0.19
    for index, probability in enumerate(PROBABILITY_COLUMNS):
        axes[0].bar(
            x + (index - 1.5) * width,
            recap_stats[probability],
            width,
            label=probability.upper(),
            color=COLORS[probability],
        )
    axes[0].set_xticks(x, recap_stats.index)
    axes[0].set_title("Category probabilities")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].bar(x, recap_stats["balance"], color=["#245d78", "#d27a35"])
    axes[1].set_xticks(x, ["Retained", "Recap\nmarked"])
    axes[1].set_title("Probability balance")
    for ax in axes:
        _clean(ax)
    fig.suptitle("Eligible assessments by recap-screen result", y=1.02)
    _save(fig, output, "prob_09_recap_comparison")

    daily_probability = (
        eligible.groupby(["issuer", "date"])
        .agg(probability_balance=("probability_balance", "mean"))
        .reset_index()
    )
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    return_panel = panel.merge(daily_probability, on=["issuer", "date"], validate="one_to_one")
    labels = ["Strong NEG", "NEG", "Near zero", "POS", "Strong POS"]
    return_panel["balance_bin"] = pd.cut(
        return_panel["probability_balance"],
        bins=[-1.001, -0.5, -0.1, 0.1, 0.5, 1.001],
        labels=labels,
    )
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), sharey=True)
    for ax, period, frame in [
        (axes[0], "Full", return_panel),
        (axes[1], "2026", return_panel[return_panel["date"].dt.year.eq(2026)]),
    ]:
        stats = (
            frame.groupby("balance_bin", observed=False)
            .agg(
                before=("adjusted_prior7", lambda value: 100 * value.mean()),
                after=("adjusted_future7", lambda value: 100 * value.mean()),
            )
            .reindex(labels)
        )
        x = np.arange(len(stats))
        ax.plot(x, stats["before"], marker="o", color="#245d78", label="Before date")
        ax.plot(x, stats["after"], marker="o", color="#d27a35", label="After date")
        ax.axhline(0, color="#334155", linewidth=0.8)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_title(period)
        _clean(ax)
    axes[0].set_ylabel("Mean adjusted seven-day return (percentage points)")
    axes[0].legend(frameon=False)
    fig.suptitle("Descriptive adjusted returns by probability balance", y=1.02)
    _save(fig, output, "prob_10_return_bins")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "articles": args.input_dir / "articles.csv",
        "pairs": args.input_dir / "sentiment_pairs.csv",
        "panel": args.input_dir / "analysis_panel.csv",
        "coefficients": args.input_dir / "period_coefficients.csv",
        "recap": args.input_dir / "recap_screen.csv",
        "validation": args.input_dir / "identity_validation.csv",
    }
    articles = pd.read_csv(files["articles"], parse_dates=["date"])
    pairs = pd.read_csv(files["pairs"], parse_dates=["date"])
    panel = pd.read_csv(files["panel"], parse_dates=["date"])
    article_lengths(articles, args.output_dir)
    period_coefficients(pd.read_csv(files["coefficients"]), args.output_dir)
    corpus_overview(articles, pairs, args.output_dir)
    validation_figure(pd.read_csv(files["validation"]), args.output_dir)
    probability_figures(pairs, articles, panel, pd.read_csv(files["recap"]), args.output_dir)
    manifest = {
        "inputs": {name: sha256(path) for name, path in files.items()},
        "outputs": {path.name: sha256(path) for path in sorted(args.output_dir.glob("*.png"))},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
