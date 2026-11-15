import argparse
from pathlib import Path

from alcv.reporting.benchmark import build_report


def main():
    parser = argparse.ArgumentParser(description="Summarize completed active learning runs")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        path = build_report(args.input_dir, args.output_dir)
    except ValueError as error:
        parser.error(str(error))
    print(f"Saved summary.csv, summary.md and learning_curves.png to {path}")


if __name__ == "__main__":
    main()
