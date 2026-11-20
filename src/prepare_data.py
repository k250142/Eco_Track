"""Prepare the raw data subset in data/raw/.

Two modes:

1. Real data (recommended for the final submission)
   python src/prepare_data.py --uci path/to/household_power_consumption.txt --start 2007-01-01 --end 2007-02-28
   Cuts a small date range out of the full UCI file and saves it as
   data/raw/household_power_subset.csv.

2. Synthetic sample (so the project runs before the UCI file is downloaded)
   python src/prepare_data.py --sample
   Writes a SYNTHETIC file in exactly the UCI format. It is NOT real household data.
"""

import argparse
import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import RAW_DIR  # noqa: E402

OUTPUT_FILE = RAW_DIR / "household_power_subset.csv"
HEADER = ["Date", "Time", "Global_active_power", "Global_reactive_power", "Voltage",
          "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]


def extract_uci_subset(source, start, end, output=OUTPUT_FILE):
    """Copy only rows between start and end (inclusive dates) from the full UCI file."""
    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date()
    kept = 0
    with open(source, newline="", encoding="utf-8") as src, open(output, "w", newline="", encoding="utf-8") as out:
        reader = csv.reader(src, delimiter=";")
        writer = csv.writer(out, delimiter=";")
        writer.writerow(next(reader))  # header
        for row in reader:
            try:
                day = datetime.strptime(row[0], "%d/%m/%Y").date()
            except (ValueError, IndexError):
                continue
            if start_date <= day <= end_date:
                writer.writerow(row)
                kept += 1
    print(f"Saved {kept} rows ({start} to {end}) to {output}")


def generate_sample(days=59, start="2007-01-01", seed=42, output=OUTPUT_FILE):
    """Write a synthetic UCI-format file with realistic daily patterns and a few bad rows."""
    rng = np.random.default_rng(seed)
    first = datetime.strptime(start, "%Y-%m-%d")
    n = days * 24 * 60
    stamps = [first + timedelta(minutes=i) for i in range(n)]
    minute_of_day = np.tile(np.arange(24 * 60), days)
    hour = minute_of_day / 60.0
    weekday = np.array([t.weekday() for t in stamps])
    weekend = (weekday >= 5).astype(float)

    def bump(center, width):
        return np.exp(-0.5 * ((hour - center) / width) ** 2)

    kw = (0.25
          + 0.9 * bump(7.5, 1.0) * (1 - 0.4 * weekend)
          + 1.6 * bump(19.5, 1.8)
          + 0.7 * bump(13.0, 2.0) * weekend
          + 0.4 * bump(23.0, 1.5) * 0.0)
    noise = np.zeros(n)
    for i in range(1, n):  # smooth random fluctuation
        noise[i] = 0.9 * noise[i - 1] + rng.normal(0, 0.05)
    spikes = (rng.random(n) < 0.004) * rng.uniform(1.0, 3.0, n)  # kettle, oven, iron...
    kw = np.clip(kw + noise + spikes, 0.1, None)

    total_wh = kw * 1000 / 60
    sub3 = total_wh * rng.uniform(0.15, 0.35, n) * (hour > 17) * 1.0
    sub1 = (rng.random(n) < 0.01) * rng.uniform(10, 35, n)
    sub2 = (rng.random(n) < 0.006) * rng.uniform(10, 30, n)
    sub_sum = np.minimum(sub1 + sub2 + sub3, total_wh)
    scale = np.divide(sub_sum, sub1 + sub2 + sub3, out=np.ones(n), where=(sub1 + sub2 + sub3) > 0)
    sub1, sub2, sub3 = np.round(sub1 * scale), np.round(sub2 * scale), np.round(sub3 * scale)

    voltage = 240 + rng.normal(0, 1.5, n)
    intensity = kw * 1000 / voltage
    reactive = np.clip(0.1 + rng.normal(0, 0.03, n), 0, None)

    missing = np.zeros(n, dtype=bool)
    missing[rng.random(n) < 0.0015] = True                 # scattered gaps
    outage_start = 20 * 24 * 60 + 5 * 60
    missing[outage_start:outage_start + 4 * 60] = True     # one 4-hour outage

    bad_rows = {1500: "abc", 9000: "-3.2", 30000: "x1"}    # malformed values
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(HEADER)
        for i, stamp in enumerate(stamps):
            date_text, time_text = stamp.strftime("%d/%m/%Y"), stamp.strftime("%H:%M:%S")
            if missing[i]:
                writer.writerow([date_text, time_text] + ["?"] * 7)
                continue
            row = [date_text, time_text, f"{kw[i]:.3f}", f"{reactive[i]:.3f}", f"{voltage[i]:.2f}",
                   f"{intensity[i]:.1f}", int(sub1[i]), int(sub2[i]), int(sub3[i])]
            if i in bad_rows:
                row[2] = bad_rows[i]
            writer.writerow(row)
        writer.writerow(["31/02/2007", "10:00:00", "1.0", "0.1", "240", "4", "0", "0", "0"])  # bad date
    print(f"Wrote synthetic sample with {n} minute rows to {output}")


def main():
    parser = argparse.ArgumentParser(description="Prepare EcoTrack raw data subset.")
    parser.add_argument("--sample", action="store_true", help="generate a synthetic UCI-format sample")
    parser.add_argument("--uci", help="path to the full UCI household_power_consumption.txt")
    parser.add_argument("--start", default="2007-01-01", help="first date to keep (YYYY-MM-DD)")
    parser.add_argument("--end", default="2007-02-28", help="last date to keep (YYYY-MM-DD)")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.uci:
        if not Path(args.uci).exists():
            sys.exit(f"File not found: {args.uci}")
        extract_uci_subset(args.uci, args.start, args.end)
    elif args.sample:
        generate_sample()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
