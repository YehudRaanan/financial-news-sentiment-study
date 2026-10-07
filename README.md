# Financial-news sentiment and stock returns

This repository contains the final report, its 15 figures, and code for the study path.

## Final outputs

- [Final report (PDF)](report/financial_news_sentiment_report.pdf)
- [Final report (Word)](report/financial_news_sentiment_report.docx)
- [Final figures](figures/)

## Exact local example

The included data are synthetic. They test the complete analysis and graph code without publishing the research data.

```powershell
git clone https://github.com/YehudRaanan/financial-news-sentiment-study.git
cd financial-news-sentiment-study
uv sync --frozen --all-extras
uv run python examples/make_example_data.py
uv run pytest
uv run python examples/reproduce_example.py
```

The test suite checks that the synthetic inputs rebuild byte for byte, validates the four saved probabilities, runs the regressions, and creates all 15 graph files.

GitHub Actions repeats these checks on Windows and Linux. The saved local smoke-test result is in
[validation/manifest.json](validation/manifest.json). It confirms successful small live calls to
GDELT, three article pages, ReFinED, one Jev article-company pair, and Yahoo Finance for two tickers.

## Full study path

The full path has five stages:

1. Discover GDELT records and extract article text.
2. Link articles to companies with ReFinED.
3. assess each article-company pair with Jev 1.13.0.
4. Build the company-date return panel and fit the fixed-effect regressions.
5. Create the final figures from the frozen tables.

The commands and data contracts are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and [DATA_CONTRACTS.md](DATA_CONTRACTS.md).

## Reproducibility claim

The synthetic example is exactly reproducible with the locked software environment. The analysis and graph code is deterministic for fixed input files.

An exact new copy of the historical corpus cannot be guaranteed from live sources. Web pages, GDELT records, Yahoo Finance data, and hosted model services can change or become unavailable. The research inputs are not in this repository. A user who has the frozen inputs can verify their checksums and reproduce the analysis and figures. A user who collects the sources again creates a new study run.

Therefore, a clone can run the locked synthetic example and all offline tests exactly. It can also
run the full source path when the user supplies Google Cloud and TypeSafe credentials and the live
services remain available. A live rerun validates the method and software path; it does not recreate
the historical corpus byte for byte.
