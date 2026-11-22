"""EnergyAnalyzer: Pandas summaries and a vectorized 2-D NumPy analysis (Labs 07, 08)."""

import pandas as pd


class EnergyAnalyzer:
    """Perform numerical calculations and consumption analysis on hourly kWh data."""

    EVENING_HOURS = (18, 22)   # slice [18:22] -> 18:00 to 21:59
    NIGHT_HOURS = (0, 6)       # slice [0:6]   -> 00:00 to 05:59

    def __init__(self, hourly_df, high_percentile=90):
        """Build the analyzer.

        Args:
            hourly_df: DataFrame indexed by hour with an 'energy_kwh' column.
            high_percentile: Percentile above which an hour counts as high usage.
        """
        if not 50 <= high_percentile < 100:
            raise ValueError("high_percentile must be between 50 and 100.")
        self._high_percentile = high_percentile
        # Pandas cleaning step: remove hours with missing energy values.
        self._df = hourly_df.dropna(subset=["energy_kwh"]).copy()
        self._df["hour_of_day"] = self._df.index.hour
        self._df["day_of_week"] = self._df.index.dayofweek

    @property
    def data(self):
        return self._df

    # ---- Pandas: filtering, grouping, summarizing --------------------------
    def hourly_summary(self):
        """Descriptive statistics of hourly kWh."""
        return self._df["energy_kwh"].describe()

    def daily_summary(self):
        """Per-day total, mean, peak and number of valid hours (grouped by day)."""
        daily = self._df["energy_kwh"].resample("D").agg(["sum", "mean", "max", "count"])
        daily.columns = ["total_kwh", "mean_kwh_per_hour", "peak_hour_kwh", "hours_measured"]
        daily = daily[daily["hours_measured"] > 0]
        daily["complete_day"] = daily["hours_measured"] == 24
        return daily

    def monthly_summary(self):
        """Per-month total, mean, peak and number of valid hours."""
        monthly = self._df["energy_kwh"].resample("MS").agg(["sum", "mean", "max", "count"])
        monthly.columns = ["total_kwh", "mean_kwh_per_hour", "peak_hour_kwh", "hours_measured"]
        monthly["mean_kwh_per_day"] = monthly["total_kwh"] / (monthly["hours_measured"] / 24.0)
        return monthly

    def hour_of_day_profile(self):
        """Average kWh for each hour of the day (0-23)."""
        return self._df.groupby("hour_of_day")["energy_kwh"].mean()

    def evening_records(self):
        """Filtered DataFrame containing only evening hours (18:00-21:59)."""
        start, end = self.EVENING_HOURS
        return self._df[(self._df["hour_of_day"] >= start) & (self._df["hour_of_day"] < end)]

    def high_usage_threshold(self):
        """kWh value at the chosen high-usage percentile."""
        return float(self._df["energy_kwh"].quantile(self._high_percentile / 100.0))

    def high_usage_periods(self, top_n=10):
        """Find the highest-usage hours.

        Returns:
            (threshold, top_list, high_hours_of_day)
            top_list: list of (timestamp, kWh) tuples sorted high to low.
            high_hours_of_day: set of clock hours whose average is above the threshold.
        """
        threshold = self.high_usage_threshold()
        high = self._df[self._df["energy_kwh"] >= threshold]
        pairs = list(zip(high.index, high["energy_kwh"]))
        top_list = sorted(pairs, key=lambda pair: pair[1], reverse=True)[:top_n]  # lambda (Lab 04)
        profile = self.hour_of_day_profile()
        high_hours = set(profile[profile >= self._df["energy_kwh"].mean() * 1.3].index)
        return threshold, top_list, high_hours
