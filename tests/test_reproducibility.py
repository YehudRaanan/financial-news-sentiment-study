from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from examples.make_example_data import build
from financial_news_sentiment.analysis import estimate_periods
from financial_news_sentiment.contracts import sha256, validate_probabilities
from financial_news_sentiment.figures import main as figures_main

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "data"


def test_example_inputs_rebuild_byte_for_byte(tmp_path: Path) -> None:
    rebuilt = tmp_path / "data"
    build(rebuilt)
    expected = sorted(path.name for path in EXAMPLE.glob("*.csv"))
    assert expected
    assert {name: sha256(rebuilt / name) for name in expected} == {
        name: sha256(EXAMPLE / name) for name in expected
    }


def test_probabilities_and_regressions_are_valid() -> None:
    pairs = pd.read_csv(EXAMPLE / "sentiment_pairs.csv")
    validate_probabilities(pairs)
    panel = pd.read_csv(EXAMPLE / "analysis_panel.csv", parse_dates=["date"])
    result = estimate_periods(panel)
    assert len(result) == 12
    assert np.isfinite(result[["coefficient", "se", "low", "high"]]).all().all()


def test_all_figures_are_created(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "figures"
    monkeypatch.setattr(
        "sys.argv",
        ["fns-figures", "--input-dir", str(EXAMPLE), "--output-dir", str(output)],
    )
    figures_main()
    assert len(list(output.glob("*.png"))) == 15
    assert (output / "manifest.json").is_file()
