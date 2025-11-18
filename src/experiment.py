from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Tuple
import csv
import random
import statistics as stats

from .state import PuzzleState
from .utils import random_solvable_state
from . import search
from .heuristics import hamming, manhattan


HeuristicMap = Dict[str, Callable[[PuzzleState], int]]


# ------------------------------- Trial gen -----------------------------------

def generate_trials(n: int, rng: random.Random) -> List[PuzzleState]:
    """
    Generate `n` solvable random start states using a provided RNG.
    """
    # Function signature:
    # - n: how many random start states to generate
    # - rng: a random.Random instance (so you can control the seed for reproducibility)
    # Returns: a list of PuzzleState objects

    return [random_solvable_state(rng) for _ in range(n)]

    # Uses a list comprehension to call random_solvable_state(rng) n times.
    # Each call returns a new solvable PuzzleState.
    # The resulting list of PuzzleState objects is returned.


# ------------------------------- Single run ----------------------------------

def run_trial(start: PuzzleState, heuristic_fn: Callable[[PuzzleState], int]):
    """
    Run A* for a single (start, heuristic) pair and return the SearchResult
    produced by search.a_star(). We don't depend on a specific class path;
    we only expect the result to expose attributes used by summarize()/CSV.
    """

    return search.a_star(start, heuristic_fn)
    # Calls the A* search algorithm with the given start state and heuristic function.
    # Returns whatever SearchResult-like object search.a_star produces.


# ------------------------------ Batch runner ---------------------------------

def run_batch(trials: List[PuzzleState], heuristics: HeuristicMap):
    """
    Evaluate every heuristic on every trial. Returns a list of SearchResult objects.
    """
    results = []  # Initialize an empty list to collect all search results.
    for start in trials: # Loop over every starting state in the list of trials.
        for name, fn in heuristics.items(): # For each trial, loop over every (heuristic name, heuristic function) pair.
            res = run_trial(start, fn)
            # Run A* on this (start, heuristic) combination.
            # res is the SearchResult returned by run_trial.

            # Ensure the heuristic name is set (a_star already sets it from fn.__name__)
            # but we enforce the display name provided by the dict key if they differ.
            if res.heuristic.lower() != name.lower():
                # Create a shallow patched object (res is a dataclass in search.py)
                # Direct assignment is fine; dataclass is not frozen.
                res.heuristic = name
            results.append(res) # Append this SearchResult to the overall results list.
    return results
# Return the full list of SearchResult objects for all (trial, heuristic) pairs.


# ------------------------------ Summarization --------------------------------

def summarize(results) -> List[Dict[str, float]]:
    """
    Compute mean/stddev per heuristic for expanded nodes and runtime (seconds).
    Also reports mean solution depth to verify comparable difficulty.
    """
    by_h: Dict[str, List] = {} # Dictionary mapping heuristic name -> list of SearchResult objects using that heuristic.
    for r in results: 
        by_h.setdefault(r.heuristic, []).append(r)

        # Ensure a list exists for the heuristic name r.heuristic in by_h.
        # Then append this result r to that list.
        # setdefault(k, []) returns the existing list for k, or adds and returns [] if missing.

    rows = []
    for hname, group in by_h.items():
        expanded_vals = [r.expanded for r in group]
        time_vals = [r.runtime_s for r in group]
        depth_vals = [r.depth for r in group if r.solved]

        row = {
            "heuristic": hname,
            "n_runs": len(group),
            "mean_expanded": stats.mean(expanded_vals),
            "std_expanded": (stats.stdev(expanded_vals) if len(expanded_vals) > 1 else 0.0),
            "mean_time_s": stats.mean(time_vals),
            "std_time_s": (stats.stdev(time_vals) if len(time_vals) > 1 else 0.0),
            "mean_depth_if_solved": (stats.mean(depth_vals) if depth_vals else 0.0),
            "solve_rate": sum(1 for r in group if r.solved) / len(group),
        }
        rows.append(row)
    # Stable order: Manhattan then Hamming if both are present
    order = {name: i for i, name in enumerate(sorted([r["heuristic"] for r in rows]))}
    rows.sort(key=lambda r: order[r["heuristic"]])
    return rows


# ------------------------------- CSV writers ---------------------------------

def save_csv(results, path: str) -> None:
    """
    Write raw per-trial results to CSV.

    Columns:
    trial, heuristic, solved, depth, expanded, runtime_s, start_state
    """
    fieldnames = ["trial", "heuristic", "solved", "depth", "expanded", "runtime_s", "start_state"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        # Create a trial index per (heuristic) group for readability
        trial_idx = 0
        for i, r in enumerate(results, start=1):
            # Make a simple rolling trial id that resets for each pair if needed later
            w.writerow({
                "trial": i,
                "heuristic": r.heuristic,
                "solved": int(bool(r.solved)),
                "depth": r.depth,
                "expanded": r.expanded,
                "runtime_s": f"{r.runtime_s:.6f}",
                "start_state": " ".join(map(str, r.start_state)),
            })


def save_summary_csv(rows: List[Dict[str, float]], path: str) -> None:
    """
    Write aggregated summary stats to CSV.

    Columns:
    heuristic, n_runs, mean_expanded, std_expanded, mean_time_s, std_time_s,
    mean_depth_if_solved, solve_rate
    """
    fieldnames = [
        "heuristic",
        "n_runs",
        "mean_expanded",
        "std_expanded",
        "mean_time_s",
        "std_time_s",
        "mean_depth_if_solved",
        "solve_rate",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ------------------------------- Pretty print --------------------------------

def format_summary_table(rows: List[Dict[str, float]]) -> str:
    """
    Return a simple aligned text table (uses tabulate if present).
    """
    try:
        from tabulate import tabulate  # optional
        return tabulate(
            rows,
            headers="keys",
            floatfmt=".3f",
            tablefmt="github",
        )
    except Exception:
        # Fallback monospace table
        headers = ["heuristic", "n_runs", "mean_expanded", "std_expanded",
                   "mean_time_s", "std_time_s", "mean_depth_if_solved", "solve_rate"]
        lines = []
        lines.append(" | ".join(headers))
        lines.append("-" * (len(lines[0]) + 5))
        for r in rows:
            line = " | ".join(str(r[h]) if not isinstance(r[h], float) else f"{r[h]:.3f}" for h in headers)
            lines.append(line)
        return "\n".join(lines)


# ------------------------------ Self-test ------------------------------------

if __name__ == "__main__":
    rng = random.Random(42)
    trials = generate_trials(5, rng)
    heuristics: HeuristicMap = {"Hamming": hamming, "Manhattan": manhattan}
    results = run_batch(trials, heuristics)
    summary = summarize(results)
    print(format_summary_table(summary))