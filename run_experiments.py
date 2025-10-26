from __future__ import annotations

import argparse
import os
import random
import sys

from src.heuristics import hamming, manhattan
from src.experiment import (
    generate_trials,
    run_batch,
    summarize,
    save_csv,
    save_summary_csv,
    format_summary_table,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run 8-puzzle A* experiments with multiple heuristics."
    )
    parser.add_argument("--trials", type=int, default=100, help="Number of random solvable start states")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--out", type=str, default="docs/results.csv", help="Output CSV file for raw results")
    parser.add_argument("--summary", type=str, default="docs/summary.csv", help="Output CSV file for summary stats")

    args = parser.parse_args(argv)

    rng = random.Random(args.seed)

    heuristics = {
        "Hamming": hamming,
        "Manhattan": manhattan,
    }

    print(f"Generating {args.trials} random solvable states (seed={args.seed})…")
    trials = generate_trials(args.trials, rng)

    print("Running A* search with both heuristics…")
    results = run_batch(trials, heuristics)

    print("Computing summary statistics…")
    summary_rows = summarize(results)

    # ensure output directory exists
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)

    save_csv(results, args.out)
    save_summary_csv(summary_rows, args.summary)

    print("\n=== Experiment Summary ===")
    print(format_summary_table(summary_rows))
    print(f"\nRaw results saved to: {args.out}")
    print(f"Summary saved to:     {args.summary}")
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())