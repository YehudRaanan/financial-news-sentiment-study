"""Rebuild the synthetic example analysis and all 15 figures."""

from __future__ import annotations

import argparse
from pathlib import Path

from financial_news_sentiment.figures import main as figures_main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/example"))
    args = parser.parse_args()
    # The figure command has its own tested CLI. Reuse it to keep one code path.
    import sys

    old_argv = sys.argv
    try:
        sys.argv = [
            "fns-figures",
            "--input-dir",
            str(Path(__file__).parent / "data"),
            "--output-dir",
            str(args.output),
        ]
        figures_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
