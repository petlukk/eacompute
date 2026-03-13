#!/usr/bin/env python3
"""Extract hypothesis and unified diffs from Loop A agent output.

Looks for HYPOTHESIS: line and FILE: + ```diff blocks.
Applies diffs with git apply. Reverts on failure.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path


def extract(text):
    """Return (hypothesis, [(filepath, diff_content), ...]) or raise ValueError."""
    hyp_match = re.search(r"^HYPOTHESIS:\s*(.+)$", text, re.MULTILINE)
    if not hyp_match:
        raise ValueError("No HYPOTHESIS: line found")
    hypothesis = hyp_match.group(1).strip()

    # Find all FILE: + diff blocks
    file_pattern = re.compile(
        r"^FILE:\s*(.+?)\s*$\n```diff\s*\n(.*?)```",
        re.MULTILINE | re.DOTALL,
    )
    matches = file_pattern.findall(text)
    if not matches:
        raise ValueError("No FILE: + diff blocks found")

    diffs = []
    for filepath, diff_content in matches:
        filepath = filepath.strip()
        diff_content = diff_content.strip() + "\n"
        if len(diff_content.strip()) < 5:
            raise ValueError(f"Diff for {filepath} too short")
        diffs.append((filepath, diff_content))

    return hypothesis, diffs


def apply_diffs(diffs, repo_root):
    """Apply all diffs. Returns None on success, error string on failure."""
    # Combine all diffs into one patch
    combined = ""
    for filepath, content in diffs:
        # Ensure diff has proper a/b prefix
        if not content.startswith("---"):
            combined += f"--- a/{filepath}\n+++ b/{filepath}\n{content}"
        else:
            combined += content
        if not combined.endswith("\n"):
            combined += "\n"

    # Write to temp file and apply
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
        f.write(combined)
        patch_path = f.name

    result = subprocess.run(
        ["git", "apply", "--allow-empty", patch_path],
        capture_output=True, text=True, cwd=repo_root,
    )

    Path(patch_path).unlink()

    if result.returncode != 0:
        return f"git apply failed: {result.stderr.strip()}"
    return None


def main():
    if len(sys.argv) < 4:
        print("Usage: parse_agent_output.py <input> <repo_root> <hypothesis.txt>",
              file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    repo_root = Path(sys.argv[2])
    hyp_path = Path(sys.argv[3])

    text = input_path.read_text()

    try:
        hypothesis, diffs = extract(text)
    except ValueError as e:
        print(f"Parse error: {e}", file=sys.stderr)
        sys.exit(1)

    # Apply diffs
    error = apply_diffs(diffs, repo_root)
    if error:
        print(f"Apply error: {error}", file=sys.stderr)
        sys.exit(1)

    hyp_path.write_text(hypothesis + "\n")
    files_changed = [f for f, _ in diffs]
    print(f"Applied {len(diffs)} diffs to: {', '.join(files_changed)}, "
          f"hypothesis: {hypothesis[:60]}...",
          file=sys.stderr)


if __name__ == "__main__":
    main()
