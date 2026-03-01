#!/usr/bin/env python3
"""
Generate test CSV files for Eastat benchmarking.

Usage:
    python generate_test.py                  # 10k rows (default)
    python generate_test.py --rows=1000000   # 1M rows
    python generate_test.py --size=100MB     # target file size
    python generate_test.py --stress         # adversarial CSV (10k rows)
    python generate_test.py --stress --rows=100000
"""

import argparse
import os
import random
import string
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Columns: id (int), name (string), price (float), quantity (int), category (string), score (float)
NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Heidi",
    "Ivan", "Judy", "Karl", "Laura", "Mallory", "Niaj", "Oscar", "Peggy",
    "Quentin", "Ruth", "Sybil", "Trent", "Ursula", "Victor", "Wendy",
    "Xander", "Yvonne", "Zelda",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
]

CATEGORIES = [
    "Electronics", "Clothing", "Food", "Books", "Sports", "Home", "Garden",
    "Automotive", "Health", "Beauty", "Toys", "Office", "Music", "Movies",
]

NULL_RATE = 0.001


def generate_row(row_id):
    """Generate a single CSV row."""
    name = random.choice(NAMES) + " " + random.choice(LAST_NAMES)
    price = round(random.uniform(0.01, 99999.99), 2)
    quantity = random.randint(0, 10000)
    category = random.choice(CATEGORIES)
    score = round(random.uniform(0.0, 100.0), 4)

    if random.random() < NULL_RATE:
        name = ""
    if random.random() < NULL_RATE:
        price_str = ""
    else:
        price_str = f"{price}"
    if random.random() < NULL_RATE:
        quantity_str = ""
    else:
        quantity_str = str(quantity)
    if random.random() < NULL_RATE:
        score_str = ""
    else:
        score_str = f"{score}"

    # Occasionally add quoted fields with commas
    if random.random() < 0.01:
        name = f'"Last, First: {name}"'

    return f"{row_id},{name},{price_str},{quantity_str},{category},{score_str}\n"


# ---------------------------------------------------------------------------
# Stress-test generation — adversarial CSV patterns
# ---------------------------------------------------------------------------

# Patterns that break naive parsers: quoted fields with embedded commas,
# newlines, doubled quotes; empty fields; BOM; \r\n; varying field lengths;
# large/small numeric values; whitespace padding.

STRESS_DESCRIPTIONS = [
    'Widget, standard',
    'Gadget "Pro" edition',
    'Cable (6ft, braided)',
    'Bolt 1/4"-20 x 2"',
    'Label: "FRAGILE"',
    'O-ring — 3mm',
    'Pipe\nfitting\nassembly',
    'Bracket [heavy duty]',
    '',
    'Tab\there',
    '   leading spaces',
    'trailing spaces   ',
    ' both sides ',
]

STRESS_NOTES = [
    'OK',
    '',
    'Ships in 2-3 days, maybe longer',
    '"Quoted note"',
    'Note with "embedded" quotes',
    'Line one\nLine two',
    'Comma, in, note',
    'a' * 500,  # long field
    'a',         # short field
    '"Has ""doubled"" quotes inside"',
    '\r\n mixed endings inside quote',
]


def generate_stress_row(row_id, line_ending):
    """Generate a single adversarial CSV row.

    Columns: id, description, amount, qty, notes, score
    Same schema shape as the clean generator but with nasty content.
    """
    # --- description: string with embedded commas, quotes, newlines ---
    desc = random.choice(STRESS_DESCRIPTIONS)
    needs_quote = any(c in desc for c in (',', '"', '\n', '\r'))
    if needs_quote:
        desc = '"' + desc.replace('"', '""') + '"'

    # --- amount: float with extreme ranges ---
    r = random.random()
    if r < 0.02:
        amount_str = ''  # null
    elif r < 0.07:
        amount_str = '0'
    elif r < 0.12:
        amount_str = str(round(random.uniform(-99999.99, -0.01), 2))  # negative
    elif r < 0.17:
        amount_str = str(round(random.uniform(1e6, 1e9), 2))  # large
    elif r < 0.22:
        amount_str = str(round(random.uniform(0.00001, 0.001), 8))  # tiny
    elif r < 0.25:
        amount_str = f'  {round(random.uniform(1, 1000), 2)}  '  # whitespace padding
    else:
        amount_str = str(round(random.uniform(0.01, 99999.99), 2))

    # --- qty: int, sometimes empty or negative ---
    r = random.random()
    if r < 0.03:
        qty_str = ''
    elif r < 0.06:
        qty_str = '0'
    elif r < 0.10:
        qty_str = str(-random.randint(1, 100))
    else:
        qty_str = str(random.randint(1, 10000))

    # --- notes: string with embedded newlines, quotes, varying length ---
    note = random.choice(STRESS_NOTES)
    if not note.startswith('"'):
        needs_quote = any(c in note for c in (',', '"', '\n', '\r'))
        if needs_quote:
            note = '"' + note.replace('"', '""') + '"'

    # --- score: float, sometimes non-numeric garbage for mixed-type stress ---
    r = random.random()
    if r < 0.02:
        score_str = ''
    elif r < 0.04:
        score_str = 'N/A'  # non-numeric in numeric column
    elif r < 0.06:
        score_str = '-'
    elif r < 0.08:
        score_str = '1e3'  # scientific notation
    else:
        score_str = str(round(random.uniform(0.0, 100.0), 4))

    return f"{row_id},{desc},{amount_str},{qty_str},{note},{score_str}{line_ending}"


def estimate_stress_row_size(line_ending):
    """Estimate average stress row size in bytes."""
    sizes = [len(generate_stress_row(i, line_ending).encode('utf-8')) for i in range(1000)]
    return sum(sizes) / len(sizes)


def generate_stress_csv(output_path, n_rows, bom=True, crlf=True, target_size=None):
    """Generate an adversarial CSV file."""
    line_ending = '\r\n' if crlf else '\n'
    header = f"id,description,amount,qty,notes,score{line_ending}"

    if target_size:
        avg_row = estimate_stress_row_size(line_ending)
        n_rows = int((target_size - len(header)) / avg_row)
        print(f"Target size: {target_size / (1024**2):.1f} MB, estimated rows: {n_rows:,}")

    print(f"Generating {n_rows:,} stress-test rows to {output_path}...")
    print(f"  BOM: {bom}, CRLF: {crlf}")
    t0 = time.perf_counter()

    with open(output_path, 'wb') as f:
        if bom:
            f.write(b'\xef\xbb\xbf')  # UTF-8 BOM
        f.write(header.encode('utf-8'))
        for i in range(1, n_rows + 1):
            f.write(generate_stress_row(i, line_ending).encode('utf-8'))
            if i % 500_000 == 0:
                elapsed = time.perf_counter() - t0
                rate = i / elapsed
                print(f"  {i:>12,} rows ({rate:,.0f} rows/s)", end='\r')

    elapsed = time.perf_counter() - t0
    file_size = os.path.getsize(output_path)
    print(f"\nDone: {n_rows:,} rows, {file_size / (1024**2):.1f} MB in {elapsed:.1f}s")

    # Print a summary of adversarial features
    print(f"\n  Adversarial features:")
    print(f"    UTF-8 BOM prefix:                   {'yes' if bom else 'no'}")
    print(f"    Line endings:                       {'\\r\\n (Windows)' if crlf else '\\n (Unix)'}")
    print(f"    Quoted fields with embedded commas:  ~8% of rows")
    print(f"    Quoted fields with embedded newlines: ~3% of rows")
    print(f"    Doubled quotes inside fields:        ~2% of rows")
    print(f"    Empty/null fields:                   ~3% per column")
    print(f"    Non-numeric in numeric column:       ~4% of score column")
    print(f"    Whitespace-padded numerics:           ~3% of amount column")
    print(f"    Very long fields (500 chars):        ~1% of notes")
    print(f"    Negative numbers:                    ~5% of amount/qty")
    print(f"    Scientific notation (1e3):           ~2% of score")
    return output_path


# ---------------------------------------------------------------------------
# Clean generation (original)
# ---------------------------------------------------------------------------

def estimate_row_size():
    """Estimate average row size in bytes."""
    sizes = [len(generate_row(i).encode()) for i in range(1000)]
    return sum(sizes) / len(sizes)


def generate_csv(output_path, n_rows, target_size=None):
    """Generate a CSV file with the specified number of rows."""
    header = "id,name,price,quantity,category,score\n"

    if target_size:
        avg_row = estimate_row_size()
        n_rows = int((target_size - len(header)) / avg_row)
        print(f"Target size: {target_size / (1024**2):.1f} MB, estimated rows: {n_rows:,}")

    print(f"Generating {n_rows:,} rows to {output_path}...")
    t0 = time.perf_counter()

    with open(output_path, 'w', buffering=1024*1024) as f:
        f.write(header)
        for i in range(1, n_rows + 1):
            f.write(generate_row(i))
            if i % 1_000_000 == 0:
                elapsed = time.perf_counter() - t0
                rate = i / elapsed
                print(f"  {i:>12,} rows ({rate:,.0f} rows/s)", end='\r')

    elapsed = time.perf_counter() - t0
    file_size = os.path.getsize(output_path)
    print(f"\nDone: {n_rows:,} rows, {file_size / (1024**2):.1f} MB in {elapsed:.1f}s")
    return output_path


def parse_size(s):
    """Parse size string like '100MB' or '1GB'."""
    s = s.upper().strip()
    multipliers = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if s.endswith(suffix):
            return int(float(s[:-len(suffix)]) * mult)
    return int(s)


def main():
    parser = argparse.ArgumentParser(description='Generate test CSV files')
    parser.add_argument('--rows', type=int, default=None, help='Number of rows')
    parser.add_argument('--size', type=str, default=None, help='Target file size (e.g., 100MB)')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--stress', action='store_true',
                        help='Generate adversarial CSV (BOM, CRLF, quoted newlines, etc.)')
    parser.add_argument('--no-bom', action='store_true', help='Stress mode: skip BOM')
    parser.add_argument('--no-crlf', action='store_true', help='Stress mode: use LF instead of CRLF')
    args = parser.parse_args()

    random.seed(args.seed)

    if args.stress:
        n_rows = args.rows or 10_000
        target = parse_size(args.size) if args.size else None
        output = args.output or str(SCRIPT_DIR / f"stress_{n_rows}.csv")
        generate_stress_csv(output, n_rows,
                            bom=not args.no_bom,
                            crlf=not args.no_crlf,
                            target_size=target)
    elif args.size:
        target = parse_size(args.size)
        output = args.output or str(SCRIPT_DIR / f"test_{args.size.lower()}.csv")
        generate_csv(output, 0, target_size=target)
    elif args.rows:
        output = args.output or str(SCRIPT_DIR / f"test_{args.rows}.csv")
        generate_csv(output, args.rows)
    else:
        output = args.output or str(SCRIPT_DIR / "test_10k.csv")
        generate_csv(output, 10_000)


if __name__ == '__main__':
    main()
