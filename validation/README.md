# Validation

`manifest.json` records the local offline and live smoke checks completed on 7 October 2026. It
contains no URL, article text, credential, raw provider response, price value, or log.

The GitHub Actions workflow repeats the offline checks on Windows and Linux for every push and pull
request. It installs all optional dependencies, imports every external adapter, runs Ruff and the
test suite, and rebuilds all 15 example figures.

Live checks confirm that each external boundary worked at the stated time. They cannot freeze a live
service. A later run can return different GDELT records, page text, Jev output, or Yahoo prices.

The manifest also records the latest successful Windows/Linux GitHub Actions run and a clean-clone
Windows run. Generated outputs remain ignored.
