# Company configuration

These files contain company metadata only. They contain no article text, model response, price
history, credential, or log.

- `company_registry.json`: 500 company identities for ReFinED title matching and the ticker fallback.
- `company_sectors.csv`: one analysis issuer and sector per company.
- `price_tickers.csv`: one deterministic Yahoo ticker per analysis issuer.
- `alias_ticker.csv`: 688 unique company-name and ticker pairs for GDELT discovery.

The files were derived from the final project registries dated 18 June and 31 August 2026 and the
500-company ReFinED benchmark registry. Source SHA-256 values:

```text
26b6f57282986414bf0d8e8c5d53ef5bc9479d9c835f0969e30c6671982d601c  company names
c9aa7f1f8b1b01a7e4f782dc97ecacd7b4b16a67b580d552af735d16aa6f0a81  company sectors
3e7bb366d553096cb10f77647373d520b6d474b0315b38536a9863fb410faefe  ReFinED benchmark registry
```

The three issuers with two listed share classes use the first ticker in sorted order, as the frozen
price code did: `GOOG`, `FOX`, and `NWS`. `BRK.B` maps to the analysis issuer `BRK.A`, following the
frozen issuer-alias rule.
