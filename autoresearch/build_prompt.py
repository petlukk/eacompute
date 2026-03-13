#!/usr/bin/env python3
"""Assemble the agent prompt from program.md, kernel, and history."""

import json
import sys
from pathlib import Path


def format_history(entries):
    """Format last N history entries as a readable summary."""
    if not entries:
        return "No previous attempts."
    lines = []
    for e in entries:
        status = "ACCEPTED" if e.get("accepted") else "REJECTED"
        time_str = f"{e['time_us']} µs" if e.get("time_us") else "N/A"
        lines.append(
            f"  #{e['iteration']}: {status} | {time_str} | "
            f"LOC {e.get('loc', '?')} | {e.get('hypothesis', '?')}"
        )
    return "\n".join(lines)


def count_loc(path):
    """Count non-blank, non-comment lines."""
    count = 0
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("//"):
                count += 1
    return count


def main():
    if len(sys.argv) < 5:
        print("Usage: build_prompt.py <program.md> <kernel.ea> <history.json> <best_score>",
              file=sys.stderr)
        sys.exit(1)

    program_path = Path(sys.argv[1])
    kernel_path = Path(sys.argv[2])
    history_path = Path(sys.argv[3])
    best_score = sys.argv[4]

    program = program_path.read_text()
    kernel = kernel_path.read_text()
    loc = count_loc(kernel_path)

    history = json.loads(history_path.read_text()) if history_path.exists() else []
    last_10 = history[-10:]

    prompt = f"""{program}

## Current Best
Score: {best_score} µs (median over 100 runs)
LOC: {loc}

## Current kernel.ea
```ea
{kernel.rstrip()}
```

## History (last {len(last_10)} attempts)
{format_history(last_10)}
"""
    print(prompt)


if __name__ == "__main__":
    main()
