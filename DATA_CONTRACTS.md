# Data contracts

The repository uses explicit tables. Full files remain local and are ignored by Git.

## `articles.csv`

One row per deduplicated article:

- `article_id`: stable article identifier
- `date`: recorded article date
- `domain`: source domain
- `words`: saved body word count
- `company_count`: linked registry companies

## `sentiment_pairs.csv`

One row per completed article-company assessment:

- `pair_id`, `article_id`, `issuer`, `date`, `domain`
- `label`: `positive`, `negative`, `mix`, or `none`
- `positive`, `negative`, `mix`, `none`: saved category probabilities, each from 0 to 1 and summing to 1 within 0.021

## `analysis_panel.csv`

One eligible row per company and date:

- `issuer`, `date`
- `positive_share`, `negative_share`, `mix_share`
- `momentum`, `volatility`, `log_past_news`
- `adjusted_prior7`, `adjusted_future7`

## `adjusted_prices.parquet`

One row per security and trading date:

- `issuer`, `date`
- `open`, `close`: Yahoo-adjusted price values from the frozen download

## `company_sectors.csv`

One row per analysis company:

- `issuer`
- `sector`

## `period_coefficients.csv`

- `target`: `prior7` or `future7`
- `period`: `full`, `earlier`, or `recent`
- `term`: `positive_share` or `negative_share`
- `coefficient`, `se`, `low`, `high`
- `n`, `issuers`, `dates`, `blocks`, `df`

## Other graph inputs

- `recap_screen.csv`: `pair_id`, `category`
- `identity_validation.csv`: `measure`, `value`
