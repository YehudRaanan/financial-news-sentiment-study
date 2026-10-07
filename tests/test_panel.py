from __future__ import annotations

import numpy as np
import pandas as pd

from financial_news_sentiment import panel


def test_exposure_history_ends_30_sessions_before_entry(monkeypatch) -> None:
    sessions = pd.bdate_range("2024-01-02", periods=320)
    prices = pd.DataFrame(
        {
            "date": sessions,
            "open": np.full(len(sessions), 100.0),
            "close": np.full(len(sessions), 101.0),
            "market_cc_return": np.full(len(sessions), 0.001),
            "sector_cc_return": np.full(len(sessions), 0.001),
            "market_oc_return": np.full(len(sessions), 0.001),
            "sector_oc_return": np.full(len(sessions), 0.001),
            "market_cc_return_peers": np.full(len(sessions), 100),
            "sector_cc_return_peers": np.full(len(sessions), 5),
        }
    )
    observed: dict[str, pd.Timestamp] = {}

    def fit(history: pd.DataFrame, _: panel.PanelSettings) -> tuple[float, float, float]:
        observed["last"] = history.index[-1]
        return 1.0, 0.0, 0.0

    monkeypatch.setattr(panel, "_fit_exposures", fit)
    article_date = sessions[270]
    value = panel._adjusted_return(prices, sessions, article_date, "future7", panel.PanelSettings())
    entry_position = sessions.get_loc(sessions[sessions > article_date][0])
    assert observed["last"] == sessions[entry_position - 30]
    assert value is not None
