#!/usr/bin/env python3
"""Assemble the agent prompt from program.md, kernel, and history.

Includes automatic bottleneck classification from benchmark GB/s data,
giving the agent direct feedback about whether compute or memory is
the limiting factor.
"""

import json
import sys
from pathlib import Path

# DRAM bandwidth range for this machine (AMD EPYC 9354P, single-thread)
DRAM_PEAK_LOW = 25.0   # GB/s — below this is definitely not bandwidth-bound
DRAM_PEAK_HIGH = 45.0  # GB/s — above this at largest size means DRAM-saturated
L2_BANDWIDTH = 80.0    # GB/s — typical L2 cache bandwidth


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


def classify_bottleneck(history_path):
    """Classify kernel bottleneck from the latest benchmark breakdown.

    Reads the most recent benchmark result's breakdown to determine:
    - DRAM-bound: largest-size GB/s near DRAM peak
    - compute-bound: largest-size GB/s well below DRAM peak
    - cache-illusion: small sizes show GB/s >> DRAM peak
    """
    bench_json = history_path.parent / "bench_result.json"
    if not bench_json.exists():
        # Try to extract from history — the benchmark output isn't saved
        # separately, but the orchestrator logs the breakdown
        return None

    try:
        data = json.loads(bench_json.read_text())
        if not data.get("breakdown"):
            return None
        return _classify_from_breakdown(data["breakdown"])
    except (json.JSONDecodeError, KeyError):
        return None


def classify_from_benchmark_output(benchmark_json_str):
    """Classify from raw benchmark JSON output string."""
    try:
        data = json.loads(benchmark_json_str)
        if not data.get("breakdown"):
            return None
        return _classify_from_breakdown(data["breakdown"])
    except (json.JSONDecodeError, KeyError):
        return None


def _classify_from_breakdown(breakdown):
    """Core classification logic from a breakdown dict."""
    sizes = list(breakdown.items())
    if not sizes:
        return None

    # Get GB/s at each size
    gbs_values = []
    for label, vals in sizes:
        gbs = vals.get("gbs")
        if gbs is not None:
            gbs_values.append((label, gbs))

    if not gbs_values:
        return None

    smallest_label, smallest_gbs = gbs_values[0]
    largest_label, largest_gbs = gbs_values[-1]

    lines = []

    # Classification
    if largest_gbs >= DRAM_PEAK_LOW:
        lines.append(
            f"⚠ BOTTLENECK: DRAM-bound ({largest_gbs:.1f} GB/s at {largest_label})"
        )
        lines.append(
            "  This kernel is at the memory bandwidth wall. "
            "Compute tricks (unrolling, accumulators, instruction reordering) "
            "will NOT help. Only reducing memory traffic can improve performance:"
        )
        lines.append(
            "  - Fuse with adjacent operations to avoid extra memory passes"
        )
        lines.append(
            "  - Use stream_store ONLY if output is write-only (not read-modify-write)"
        )
        lines.append(
            "  - Reduce array count or element size if possible"
        )
        lines.append(
            "  - If none of these apply, this kernel is already optimal."
        )
    elif largest_gbs < 10.0:
        lines.append(
            f"✓ BOTTLENECK: Compute-bound ({largest_gbs:.1f} GB/s at {largest_label}, "
            f"well below DRAM peak ~{DRAM_PEAK_LOW}-{DRAM_PEAK_HIGH} GB/s)"
        )
        lines.append(
            "  There is significant headroom for optimization. Try:"
        )
        lines.append("  - Wider SIMD (f32x4 → f32x8, u8x16 → u8x32)")
        lines.append("  - Loop unrolling for ILP")
        lines.append("  - Multiple accumulators to break dependency chains")
        lines.append("  - FMA fusion to reduce instruction count")
        lines.append("  - Algorithmic restructuring (loop reordering, tiling)")
    else:
        lines.append(
            f"◐ BOTTLENECK: Mixed ({largest_gbs:.1f} GB/s at {largest_label})"
        )
        lines.append(
            "  Between compute-bound and DRAM-bound. "
            "Some compute optimizations may help, but gains will be modest."
        )

    # Cache illusion warning
    if smallest_gbs > DRAM_PEAK_HIGH and largest_gbs < smallest_gbs * 0.5:
        lines.append("")
        lines.append(
            f"⚠ CACHE ILLUSION: {smallest_label} shows {smallest_gbs:.0f} GB/s "
            f"(L2/L3 cache), but {largest_label} drops to {largest_gbs:.0f} GB/s (DRAM). "
            f"Optimizations that only help small sizes are worthless."
        )

    # Per-size breakdown
    lines.append("")
    lines.append("  Bandwidth by size:")
    for label, gbs in gbs_values:
        tier = "L2" if gbs > L2_BANDWIDTH else "L3" if gbs > DRAM_PEAK_HIGH else "DRAM"
        lines.append(f"    {label}: {gbs:.1f} GB/s ({tier})")

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
        print("Usage: build_prompt.py <program.md> <kernel.ea> <history.json> "
              "<best_score> [benchmark_json]",
              file=sys.stderr)
        sys.exit(1)

    program_path = Path(sys.argv[1])
    kernel_path = Path(sys.argv[2])
    history_path = Path(sys.argv[3])
    best_score = sys.argv[4]
    benchmark_json = sys.argv[5] if len(sys.argv) > 5 else None

    program = program_path.read_text()
    kernel = kernel_path.read_text()
    loc = count_loc(kernel_path)

    history = json.loads(history_path.read_text()) if history_path.exists() else []
    last_10 = history[-10:]

    # Bottleneck classification
    bottleneck_section = ""
    if benchmark_json:
        classification = classify_from_benchmark_output(benchmark_json)
        if classification:
            bottleneck_section = f"\n## Bottleneck Analysis\n{classification}\n"

    prompt = f"""{program}

## Current Best
Score: {best_score} µs (largest dataset size)
LOC: {loc}
{bottleneck_section}
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
