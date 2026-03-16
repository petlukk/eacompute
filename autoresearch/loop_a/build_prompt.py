#!/usr/bin/env python3
"""Assemble the Loop A agent prompt from compiler guide, feature request, source files, and history."""

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
        error = e.get("error", "")
        error_str = f" | error: {error[:300]}" if error else ""
        lines.append(
            f"  #{e['iteration']}: {status} | {e.get('hypothesis', '?')}{error_str}"
        )
    return "\n".join(lines)


def format_baselines(baselines):
    """Format benchmark baselines as a table."""
    if not baselines:
        return "No baselines yet."
    lines = ["  Kernel       | Time (µs)"]
    lines.append("  -------------|----------")
    for name, time_us in sorted(baselines.items()):
        lines.append(f"  {name:<13}| {time_us}")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 6:
        print("Usage: build_prompt.py <program.md> <compiler_guide.md> "
              "<feature_request> <history.json> <baselines.json> [source_files...]",
              file=sys.stderr)
        sys.exit(1)

    program = Path(sys.argv[1]).read_text()
    guide = Path(sys.argv[2]).read_text()
    feature_request = sys.argv[3]
    history = json.loads(Path(sys.argv[4]).read_text()) if Path(sys.argv[4]).exists() else []
    baselines = json.loads(Path(sys.argv[5]).read_text()) if Path(sys.argv[5]).exists() else {}
    source_files = sys.argv[6:]

    last_10 = history[-10:]

    # Include relevant source files
    source_sections = []
    for sf in source_files:
        p = Path(sf)
        if p.exists():
            source_sections.append(f"## {sf}\n```rust\n{p.read_text().rstrip()}\n```")

    prompt = f"""{program}

## Compiler Guide
{guide}

## Current Feature Request
{feature_request}

## Relevant Source Files
{chr(10).join(source_sections) if source_sections else "No source files provided."}

## Benchmark Baselines
{format_baselines(baselines)}

## History (last {len(last_10)} attempts)
{format_history(last_10)}
"""
    print(prompt)


if __name__ == "__main__":
    main()
