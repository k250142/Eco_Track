"""EnergyDataManager: reading, validating, aggregating and storing energy records.

Input format: the UCI "Individual Household Electric Power Consumption" file
(semicolon separated, one row per minute, '?' marks a missing value).
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.exceptions import DataValidationError
from src.utils import PROCESSED_DIR, RAW_DIR

# Columns expected in the UCI file. Tuple: this list never changes.
NUMERIC_COLUMNS = (
    "Global_active_power",    # kW, minute average
    "Global_reactive_power",  # kVAr
    "Voltage",                # V
    "Global_intensity",       # A
    "Sub_metering_1",         # Wh in that minute (kitchen)
    "Sub_metering_2",         # Wh in that minute (laundry)
    "Sub_metering_3",         # Wh in that minute (water heater / air conditioner)
)
REQUIRED_COLUMNS = ("Date", "Time") + NUMERIC_COLUMNS

# A clock hour needs at least this many valid minutes to be kept.
MIN_MINUTES_PER_HOUR = 45


class EnergyDataManager:
    """Manage electricity records and all related file operations."""

    def __init__(self, raw_path=None, processed_dir=None):
        self._raw_path = Path(raw_path) if raw_path else RAW_DIR / "household_power_subset.csv"
        self._processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
        self._hourly = None          # DataFrame, index = hour
        self._report = {}            # dict with validation counts

    # ---- encapsulated read-only access -------------------------------------
    @property
    def raw_path(self):
        return self._raw_path

    @property
    def hourly(self):
        """Hourly DataFrame (raises if nothing is loaded yet)."""
        if self._hourly is None:
            raise DataValidationError("No data loaded yet. Use menu option 1 first.")
        return self._hourly

    @property
    def report(self):
        return dict(self._report)

    @property
    def is_loaded(self):
        return self._hourly is not None

    # ---- reading and validation (CSV) ---------------------------------------
    def load_raw(self):
        """Read the raw CSV, validate each row, and build the hourly table.

        Returns:
            The validation report (dict).
        Raises:
            DataValidationError: file missing, wrong columns, or no valid rows.
        """
        if not self._raw_path.exists():
            raise DataValidationError(
                f"Raw data file not found: {self._raw_path}\n"
                "  Run 'python src/prepare_data.py --sample' or see the README.",
                file_path=self._raw_path,
            )

        records, invalid_examples = [], []
        total = missing = malformed = 0
        try:
            with open(self._raw_path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter=";")
                absent = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
                if absent:
                    raise DataValidationError(f"Missing column(s) in file: {absent}", self._raw_path)

                for line_no, row in enumerate(reader, start=2):
                    total += 1
                    status, record = self._validate_row(row)
                    if status == "ok":
                        records.append(record)
                    elif status == "missing":
                        missing += 1
                    else:
                        malformed += 1
                        if len(invalid_examples) < 5:
                            invalid_examples.append(f"line {line_no}: {dict(row)}")
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            raise DataValidationError(f"Could not read {self._raw_path}: {exc}", self._raw_path) from exc

        if not records:
            raise DataValidationError("The file contains no valid records.", self._raw_path)

        minutes = pd.DataFrame(records).set_index("timestamp").sort_index()
        minutes = minutes[~minutes.index.duplicated(keep="first")]
        self._hourly = self._aggregate_hourly(minutes)

        self._report = {
            "raw_file": self._raw_path.name,
            "rows_read": total,
            "valid_rows": len(records),
            "rows_with_missing_values": missing,
            "malformed_rows": malformed,
            "malformed_examples": invalid_examples,
            "date_from": str(minutes.index.min()),
            "date_to": str(minutes.index.max()),
            "hours_kept": int(self._hourly["energy_kwh"].notna().sum()),
            "hours_dropped_low_coverage": int(self._hourly["energy_kwh"].isna().sum()),
        }
        return self.report

    @staticmethod
    def _validate_row(row):
        """Classify one CSV row as 'ok', 'missing' ('?' or blank) or 'malformed'."""
        try:
            timestamp = datetime.strptime(f"{row['Date']} {row['Time']}", "%d/%m/%Y %H:%M:%S")
        except (ValueError, TypeError):
            return "malformed", None

        values = {}
        for column in NUMERIC_COLUMNS:
            raw = (row.get(column) or "").strip()
            if raw in ("", "?"):
                return "missing", None
            try:
                number = float(raw)
            except ValueError:
                return "malformed", None
            if number < 0:  # power, voltage and energy cannot be negative
                return "malformed", None
            values[column] = number
        return "ok", {"timestamp": timestamp, **values}

    @staticmethod
    def _aggregate_hourly(minutes):
        """Convert per-minute kW readings into hourly energy (kWh).

        Power (kW) is NOT energy. For each clock hour we take the mean power over
        the valid minutes and multiply by 1 hour (time-weighted average), which
        equals the sum of kW x (1/60 h). Hours with fewer than MIN_MINUTES_PER_HOUR
        valid minutes are set to NaN. Sub-meter values are Wh per minute and are
        summed then converted to kWh.
        """
        grouped = minutes.resample("h")
        hourly = pd.DataFrame({
            "energy_kwh": grouped["Global_active_power"].mean() * 1.0,
            "minutes_valid": grouped["Global_active_power"].count(),
            "avg_voltage": grouped["Voltage"].mean(),
            "kitchen_kwh": grouped["Sub_metering_1"].sum() / 1000.0,
            "laundry_kwh": grouped["Sub_metering_2"].sum() / 1000.0,
            "heater_ac_kwh": grouped["Sub_metering_3"].sum() / 1000.0,
        })
        low = hourly["minutes_valid"] < MIN_MINUTES_PER_HOUR
        hourly.loc[low, ["energy_kwh", "avg_voltage", "kitchen_kwh", "laundry_kwh", "heater_ac_kwh"]] = float("nan")
        hourly.index.name = "hour"
        return hourly

    # ---- storing processed data (CSV, JSON, TXT) ---------------------------
    def save_processed(self):
        """Write processed outputs: hourly CSV, summary JSON, validation TXT.

        Returns:
            List of the paths written.
        """
        data = self.hourly
        self._processed_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self._processed_dir / "hourly_energy.csv"
        json_path = self._processed_dir / "data_summary.json"
        txt_path = self._processed_dir / "validation_report.txt"

        data.to_csv(csv_path)

        clean = data["energy_kwh"].dropna()
        summary = {
            "hours": int(len(clean)),
            "total_kwh": round(float(clean.sum()), 3),
            "mean_kwh_per_hour": round(float(clean.mean()), 4),
            "max_kwh_in_one_hour": round(float(clean.max()), 4),
        }
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump({"validation": self._report, "summary": summary}, handle, indent=2)

        lines = [
            "EcoTrack data validation report",
            "=" * 34,
            *[f"{key}: {value}" for key, value in self._report.items() if key != "malformed_examples"],
            "Malformed row examples:",
            *(self._report["malformed_examples"] or ["  none"]),
        ]
        with open(txt_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        return [csv_path, json_path, txt_path]

    def load_processed(self):
        """Load a previously saved hourly CSV. Returns True on success, False if absent."""
        csv_path = self._processed_dir / "hourly_energy.csv"
        if not csv_path.exists():
            return False
        try:
            self._hourly = pd.read_csv(csv_path, index_col="hour", parse_dates=["hour"])
        except (OSError, ValueError, KeyError) as exc:
            raise DataValidationError(f"Processed file is unreadable: {exc}", csv_path) from exc

        json_path = self._processed_dir / "data_summary.json"
        if json_path.exists():
            try:
                with open(json_path, encoding="utf-8") as handle:
                    self._report = json.load(handle).get("validation", {})
            except (OSError, json.JSONDecodeError):
                self._report = {}
        return True
