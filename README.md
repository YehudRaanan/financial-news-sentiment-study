# Financial-news sentiment and stock returns

This repository contains the final report, its 15 figures, and clean code for the final study path. It does not contain article bodies, research datasets, provider responses, credentials, logs, intermediate files, or abandoned experiments.

## Final outputs

- [Final report (PDF)](report/financial_news_sentiment_report.pdf)
- [Final report (Word)](report/financial_news_sentiment_report.docx)
- [Final figures](figures/)

## Exact local example

The included data are synthetic. They test the complete analysis and graph code without publishing the research data.

```powershell
git clone https://github.com/YehudRaanan/financial-news-sentiment-study.git
cd financial-news-sentiment-study
uv sync --frozen --extra dev
uv run python examples/make_example_data.py
uv run pytest
uv run python examples/reproduce_example.py
```

The test suite checks that the synthetic inputs rebuild byte for byte, validates the four saved probabilities, runs the regressions, and creates all 15 graph files.

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

An exact new copy of the historical corpus cannot be guaranteed from live sources. Web pages, GDELT records, Yahoo Finance data, and hosted model services can change or become unavailable. The private research inputs are also not in this repository. A user who has the frozen inputs can verify their checksums and reproduce the analysis and figures. A user who collects the sources again creates a new study run.

## Public release status

The repository starts as private. Before public release, select a license, confirm permission for every included report element, and run the checks in [PUBLIC_RELEASE_CHECKLIST.md](PUBLIC_RELEASE_CHECKLIST.md).
