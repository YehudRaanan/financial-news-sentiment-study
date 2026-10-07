# Reproducibility

## Environment

- Python 3.12.7, set in `.python-version`
- uv 0.11.1
- Dependencies are pinned in `pyproject.toml` and `uv.lock`.
- Run commands from the repository root with `uv run`.
- Full source collection needs Google Cloud credentials, a GDELT BigQuery project, network access, and a TypeSafe API key.

## Input sequence

### 1. Discover GDELT records

Prepare `aliases.csv` with `alias,ticker`. Then run:

```powershell
uv sync --frozen --extra collect
uv run fns-sources discover --aliases aliases.csv --project YOUR_GCP_PROJECT --start 2023-01-01 --end 2026-06-18 --output data/work_queue.parquet
```

The final study has no GDELT records from 15 June through 1 July 2025. Do not add the separately recovered publisher-dated records.

### 2. Extract article text

```powershell
uv run fns-sources extract --queue data/work_queue.parquet --output data/articles.parquet
```

The extractor keeps one normalized body per content hash. Live pages can change, so save the resulting file and its checksum for a frozen run.

### 3. Link companies

Use a registry JSON with `cik`, `issuer`, `canonical_name`, `tickers`, and `wikipedia_titles` for each company.

The historical run used ReFinED 1.0 at commit `7c98036f72c39a8d6d2c097bbde89ea3731901f0`, model `wikipedia_model_with_numbers`, entity set `wikipedia`, and a post-ReFinED `exchange:ticker` fallback only when ReFinED found no registry company. The command implements both stages.

```powershell
uv sync --frozen --extra identity
uv run fns-sources link --articles data/articles.parquet --registry company_registry.json --output data/article_company.parquet
```

### 4. Assess sentiment

Set `TYPESAFE_API_KEY` outside the repository. The command sends the complete title and body once per article and asks one four-category question for every linked company.

```powershell
$env:TYPESAFE_API_KEY = "..."
uv run fns-sources sentiment --articles data/articles.parquet --links data/article_company.parquet --output data/sentiment_pairs.parquet
```

The fixed model is `jev-1.13.0`. A future service response can differ from the historical response. Save the returned table and its checksum.

### 5. Build the return panel

Prepare a CSV that maps one analysis ticker to each issuer. Download prices with the saved study settings:

```powershell
uv run fns-sources prices --registry price_tickers.csv --start 2022-12-01 --end 2026-10-01 --output data/adjusted_prices.parquet
```

Then build the panel:

```powershell
uv run fns-panel --pairs data/sentiment_pairs.parquet --prices data/adjusted_prices.parquet --registry company_sectors.csv --output data/analysis_panel.parquet
```

The panel code uses:

- one row per company and recorded article date;
- directional shares with NONE excluded from the denominator;
- earlier-news volume from days 14 through 8 before the recorded date;
- momentum from the prior 21 sessions and volatility from the prior 60 sessions;
- returns adjusted with equal-weight market and sector comparison groups;
- historical exposure estimates from at most 250 sessions, ending 30 sessions before each return window, with at least 100 observations;
- complete prior 7, following 7, following 14, and following 30 calendar-day return windows;
- iterative removal of company or date groups with only one row.

The report regressions use the positive and negative shares together, the three prior controls, company effects, recorded-date effects, and equal row weights.

```powershell
uv run fns-analysis --pairs data/sentiment_pairs.parquet --panel data/analysis_panel.parquet --output outputs/period_coefficients.csv
```

### 6. Create figures

Export the contracted inputs as CSV files in one directory. Then run:

```powershell
uv run fns-figures --input-dir data/final_tables --output-dir outputs/figures
```

The command writes 15 PNG files and `manifest.json` with input and output SHA-256 checksums.

## Exact and conditional parts

| Part | Reproducibility |
|---|---|
| Synthetic example | Exact and tested |
| Regression and graph code with fixed inputs | Deterministic and tested |
| Historical outputs with the original frozen inputs | Reproducible after the private inputs are supplied and hashes match |
| New collection from live services | Method reproducible; exact historical records and responses are not guaranteed |
