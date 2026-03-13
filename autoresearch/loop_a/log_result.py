#!/usr/bin/env python3
"""Append a result entry to Loop A history.json."""

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 6:
        print("Usage: log_result.py <history.json> <iteration> <hypothesis> "
              "<accepted> <error> [benchmark_json]",
              file=sys.stderr)
        sys.exit(1)

    history_path = Path(sys.argv[1])
    iteration = int(sys.argv[2])
    hypothesis = sys.argv[3]
    accepted = sys.argv[4] == "true"
    error = sys.argv[5] if sys.argv[5] != "null" else None

    benchmarks = None
    if len(sys.argv) > 6 and sys.argv[6] != "null":
        try:
            benchmarks = json.loads(sys.argv[6])
        except json.JSONDecodeError:
            pass

    history = json.loads(history_path.read_text()) if history_path.exists() else []

    entry = {
        "iteration": iteration,
        "hypothesis": hypothesis,
        "accepted": accepted,
        "error": error,
    }
    if benchmarks:
        entry["benchmarks"] = benchmarks

    history.append(entry)
    history_path.write_text(json.dumps(history, indent=2) + "\n")

    status = "ACCEPTED" if accepted else "REJECTED"
    print(f"Logged iteration {iteration}: {status}", file=sys.stderr)


if __name__ == "__main__":
    main()
