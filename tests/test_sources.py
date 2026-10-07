from __future__ import annotations

import io
import json
from types import SimpleNamespace
from typing import Any

import pandas as pd

from financial_news_sentiment.sources import (
    MODEL,
    assess_sentiment,
    discover_gdelt,
    download_prices,
    extract_articles,
    link_companies,
)


class _BigQuery:
    class ArrayQueryParameter:
        def __init__(self, *values: Any) -> None:
            self.values = values

    ScalarQueryParameter = ArrayQueryParameter

    class QueryJobConfig:
        def __init__(self, **values: Any) -> None:
            self.__dict__.update(values)


class _BigQueryClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def query(self, sql: str, job_config: Any) -> Any:
        self.calls.append((sql, job_config))
        if getattr(job_config, "dry_run", False):
            return SimpleNamespace(total_bytes_processed=1_000)
        frame = pd.DataFrame(
            [
                {
                    "url": "https://example.test/a",
                    "domain": "reuters.com",
                    "ticker": "AAPL",
                    "date": "2026-01-02",
                }
            ]
        )
        return SimpleNamespace(to_dataframe=lambda: frame)


class _Response(io.BytesIO):
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class _Opener:
    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body
        self.request: Any | None = None

    def open(self, request: Any, timeout: int) -> _Response:
        self.request = request
        assert timeout == 60
        return _Response(json.dumps(self.body).encode())


def test_gdelt_adapter_uses_dry_run_and_billing_cap() -> None:
    client = _BigQueryClient()
    aliases = pd.DataFrame([{"alias": "apple", "ticker": "AAPL"}])
    result = discover_gdelt(
        aliases,
        "test-project",
        "2026-01-01",
        "2026-01-03",
        2_000,
        client=client,
        bigquery_module=_BigQuery,
    )
    assert len(client.calls) == 2
    assert client.calls[0][1].dry_run is True
    assert client.calls[1][1].maximum_bytes_billed == 2_000
    assert result.attrs["estimated_bytes_processed"] == 1_000
    assert result.loc[0, "ticker"] == "AAPL"
    assert "nasdaq\\-inc|ndaq" in client.calls[0][0]


def test_article_extraction_keeps_first_copy_and_unions_tickers() -> None:
    queue = pd.DataFrame(
        [
            {
                "url": "https://example.test/first",
                "domain": "reuters.com",
                "date": "2026-01-02",
                "ticker": "AAPL",
            },
            {
                "url": "https://example.test/copy",
                "domain": "cnbc.com",
                "date": "2026-01-03",
                "ticker": "MSFT",
            },
        ]
    )

    def extract(downloaded: str, **_: Any) -> Any:
        text = (
            "Apple reports higher revenue."
            if downloaded == "first"
            else " apple REPORTS higher  revenue. "
        )
        return SimpleNamespace(title="Results", text=text)

    result = extract_articles(
        queue,
        fetch_url=lambda url: "first" if url.endswith("first") else "copy",
        bare_extraction=extract,
    )
    assert len(result) == 1
    assert result.loc[0, "url"].endswith("first")
    assert json.loads(result.loc[0, "tickers_from_gdelt"]) == ["AAPL", "MSFT"]


def test_refined_adapter_and_exchange_ticker_fallback() -> None:
    registry = [
        {
            "cik": "0000320193",
            "issuer": "AAPL",
            "canonical_name": "Apple",
            "tickers": ["AAPL"],
            "wikipedia_titles": ["Apple Inc."],
        }
    ]
    articles = pd.DataFrame(
        [
            {"article_id": "a", "title": "Apple results", "body": "Apple grew."},
            {"article_id": "b", "title": "Ticker item", "body": "NASDAQ:AAPL rose."},
        ]
    )

    class Model:
        def process_text(self, text: str, **_: Any) -> list[Any]:
            if text.startswith("Ticker"):
                return []
            entity = SimpleNamespace(wikipedia_entity_title="Apple Inc.")
            return [SimpleNamespace(text="Apple", predicted_entity=entity)]

    result = link_companies(articles, registry, model=Model())
    assert result["article_id"].tolist() == ["a", "b"]
    assert result["method"].tolist() == [
        "refined_aida",
        "exchange_ticker_after_refined_no_registry_company",
    ]


def test_jev_adapter_validates_full_choice_response() -> None:
    articles = pd.DataFrame(
        [
            {
                "article_id": "a",
                "title": "Results",
                "body": "Apple increased revenue.",
                "date": "2026-01-02",
                "domain": "reuters.com",
            }
        ]
    )
    links = pd.DataFrame(
        [
            {
                "article_id": "a",
                "cik": "0000320193",
                "issuer": "AAPL",
                "canonical_name": "Apple",
                "tickers": '["AAPL"]',
            }
        ]
    )
    probabilities = {"positive": 0.7, "negative": 0.1, "mix": 0.1, "none": 0.1}
    opener = _Opener(
        {
            "model": MODEL,
            "answers": {
                "c0000320193": {
                    "type": "choice",
                    "choice": "positive",
                    "probabilities": probabilities,
                    "confidence": 0.6,
                }
            },
            "usage": {"input_tokens": 120, "output_tokens": 20},
        }
    )
    result = assess_sentiment(articles, links, "test-key", opener=opener)
    assert result.loc[0, "label"] == "positive"
    assert result.loc[0, "confidence"] == 0.6
    assert result.loc[0, "input_tokens"] == 120
    sent = json.loads(opener.request.data)
    assert sent["model"] == MODEL
    assert set(sent["questions"]["c0000320193"]["criteria"]) == set(probabilities)


def test_yahoo_adapter_uses_frozen_download_settings() -> None:
    calls: list[dict[str, Any]] = []

    def download(ticker: str, **settings: Any) -> pd.DataFrame:
        calls.append({"ticker": ticker, **settings})
        return pd.DataFrame(
            {"Open": [100.0, 101.0], "Close": [101.0, 102.0]},
            index=pd.DatetimeIndex(["2026-01-02", "2026-01-05"], name="Date"),
        )

    registry = pd.DataFrame(
        [{"issuer": "AAPL", "ticker": "AAPL"}, {"issuer": "MSFT", "ticker": "MSFT"}]
    )
    result = download_prices(registry, "2026-01-01", "2026-01-06", download=download)
    assert len(result) == 4
    assert {call["ticker"] for call in calls} == {"AAPL", "MSFT"}
    assert all(call["auto_adjust"] and call["repair"] for call in calls)
    assert all(call["actions"] is False and call["progress"] is False for call in calls)
