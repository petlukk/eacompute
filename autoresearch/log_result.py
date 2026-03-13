#!/usr/bin/env python3
"""Append a result entry to history.json."""

import hashlib
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 8:
        print("Usage: log_result.py <history.json> <iteration> <hypothesis> "
              "<time_us> <loc> <correct> <accepted> [kernel.ea]",
              file=sys.stderr)
        sys.exit(1)

    history_path = Path(sys.argv[1])
    iteration = int(sys.argv[2])
    hypothesis = sys.argv[3]
    time_us = float(sys.argv[4]) if sys.argv[4] != "null" else None
    loc = int(sys.argv[5]) if sys.argv[5] != "null" else None
    correct = sys.argv[6] == "true"
    accepted = sys.argv[7] == "true"

    kernel_hash = None
    if len(sys.argv) > 8:
        kernel_path = Path(sys.argv[8])
        if kernel_path.exists():
            kernel_hash = hashlib.sha256(kernel_path.read_bytes()).hexdigest()[:12]

    history = json.loads(history_path.read_text()) if history_path.exists() else []

    history.append({
        "iteration": iteration,
        "hypothesis": hypothesis,
        "time_us": time_us,
        "loc": loc,
        "correct": correct,
        "accepted": accepted,
        "kernel_hash": kernel_hash,
    })

    history_path.write_text(json.dumps(history, indent=2) + "\n")
    print(f"Logged iteration {iteration}: "
          f"{'ACCEPTED' if accepted else 'REJECTED'} "
          f"({time_us} µs, LOC {loc})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
