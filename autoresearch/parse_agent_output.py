#!/usr/bin/env python3
"""Extract hypothesis and kernel source from agent output.

Looks for HYPOTHESIS: line and first code fence block.
Only writes output files on successful extraction.
"""

import re
import sys
from pathlib import Path


def extract(text):
    """Return (hypothesis, kernel_source) or raise ValueError."""
    hyp_match = re.search(r"^HYPOTHESIS:\s*(.+)$", text, re.MULTILINE)
    if not hyp_match:
        raise ValueError("No HYPOTHESIS: line found")
    hypothesis = hyp_match.group(1).strip()

    # Prefer ```ea fences, fall back to bare ```
    fence_match = re.search(r"```ea\s*\n(.*?)```", text, re.DOTALL)
    if not fence_match:
        fence_match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if not fence_match:
        raise ValueError("No code fence block found")
    kernel = fence_match.group(1).strip() + "\n"

    if len(kernel.strip()) < 10:
        raise ValueError(f"Kernel too short ({len(kernel.strip())} chars)")

    return hypothesis, kernel


def main():
    if len(sys.argv) < 4:
        print("Usage: parse_agent_output.py <input> <kernel.ea> <hypothesis.txt>",
              file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    kernel_path = Path(sys.argv[2])
    hyp_path = Path(sys.argv[3])

    text = input_path.read_text()

    try:
        hypothesis, kernel = extract(text)
    except ValueError as e:
        print(f"Parse error: {e}", file=sys.stderr)
        sys.exit(1)

    kernel_path.write_text(kernel)
    hyp_path.write_text(hypothesis + "\n")
    print(f"Extracted: {len(kernel)} bytes, hypothesis: {hypothesis[:60]}...",
          file=sys.stderr)


if __name__ == "__main__":
    main()
