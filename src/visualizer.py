"""EnergyVisualizer: labelled Matplotlib / Seaborn charts saved as PNG (Lab 09)."""

import matplotlib

matplotlib.use("Agg")  # save to files; never opens a blocking window
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from src.utils import FIGURES_DIR  # noqa: E402

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


class EnergyVisualizer:
    """Create charts from the hourly energy data."""

    def __init__(self, analyzer, output_dir=None):
        self._analyzer = analyzer
        self._dir = output_dir or FIGURES_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def _save(self, fig, filename):
        path = self._dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path

    def consumption_over_time(self):
        """Chart 1: hourly consumption with the daily average overlaid."""
        df = self._analyzer.data["energy_kwh"]
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(df.index, df.values, color="#9ecae1", linewidth=0.6, label="Hourly consumption")
        daily_mean = df.resample("D").mean()
        ax.plot(daily_mean.index, daily_mean.values, color="#08519c", linewidth=2, label="Daily average")
        ax.set_title("Household Electricity Consumption Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Energy (kWh per hour)")
        ax.legend()
        return self._save(fig, "01_consumption_over_time.png")

    def usage_by_weekday(self):
        """Chart 2: average total energy per day of the week (complete days only)."""
        daily = self._analyzer.daily_summary()
        daily = daily[daily["complete_day"]]
        by_day = daily.groupby(daily.index.dayofweek)["total_kwh"].mean().reindex(range(7))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(x=[DAY_NAMES[i] for i in by_day.index], y=by_day.values, color="#3182bd", ax=ax)
        ax.set_title("Average Daily Energy Use by Day of Week")
        ax.set_xlabel("Day of week")
        ax.set_ylabel("Average energy per day (kWh)")
        return self._save(fig, "02_usage_by_weekday.png")

    def usage_distribution(self):
        """Chart 3: histogram of hourly consumption with median and high-usage line."""
        values = self._analyzer.data["energy_kwh"]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.histplot(values, bins=40, kde=True, color="#6baed6", ax=ax)
        ax.axvline(values.median(), color="green", linestyle="--", label=f"Median ({values.median():.2f} kWh)")
        threshold = self._analyzer.high_usage_threshold()
        ax.axvline(threshold, color="red", linestyle="--", label=f"High-usage threshold ({threshold:.2f} kWh)")
        ax.set_title("Distribution of Hourly Energy Consumption")
        ax.set_xlabel("Energy (kWh per hour)")
        ax.set_ylabel("Number of hours")
        ax.legend()
        return self._save(fig, "03_usage_distribution.png")

    def create_all_basic(self):
        """Create the data charts and return their paths."""
        return [self.consumption_over_time(), self.usage_by_weekday(),
                self.usage_distribution()]
