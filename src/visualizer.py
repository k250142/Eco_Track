"""EnergyVisualizer: labelled Matplotlib / Seaborn charts saved as PNG (Lab 09)."""

import matplotlib

matplotlib.use("Agg")  # save to files; never opens a blocking window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
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

    def hourly_profile(self):
        """Chart 4: average hour-of-day profile, weekdays versus weekends."""
        result = self._analyzer.numpy_day_hour_analysis()
        hours = np.arange(24)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(hours, result["weekday_profile"], marker="o", label="Weekdays")
        ax.plot(hours, result["weekend_profile"], marker="s", label="Weekends")
        ax.set_xticks(range(0, 24, 2))
        ax.set_title("Average Daily Load Profile: Weekdays vs Weekends")
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Average energy (kWh per hour)")
        ax.legend()
        return self._save(fig, "04_hourly_profile.png")

    def confusion_matrices(self, results):
        """Chart 5: confusion matrices of the compared models (from ml_models results)."""
        fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
        axes = np.atleast_1d(axes)
        for ax, (name, info) in zip(axes, results.items()):
            sns.heatmap(np.array(info["confusion_matrix"]), annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=["Low", "High"], yticklabels=["Low", "High"], ax=ax)
            ax.set_title(f"{name}\n(accuracy {info['accuracy']:.2f})")
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("Actual label")
        return self._save(fig, "05_confusion_matrices.png")

    def create_all_basic(self):
        """Create the four data charts and return their paths."""
        return [self.consumption_over_time(), self.usage_by_weekday(),
                self.usage_distribution(), self.hourly_profile()]
