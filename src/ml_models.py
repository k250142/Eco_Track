"""Machine learning: Logistic Regression vs Decision Tree on the same chronological split (Labs 10-11).

Task: predict whether the NEXT hour will be a high- or low-consumption hour.

Leakage rules followed here:
  * The split is chronological (first 80% of hours = train, last 20% = test), never shuffled.
  * The high/low threshold is the median of the TRAINING hours only, then reused for the test data.
  * Features at time t use only information up to hour t (lags, rolling mean of past hours,
    calendar features). The target is the label of hour t+1.
  * A training row is dropped if its target hour falls in the test period.
  * Scaling (for Logistic Regression) is fitted on the training rows only (inside a Pipeline).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURE_NAMES = ["lag_1h", "lag_2h", "lag_3h", "lag_24h", "rolling_mean_3h",
                 "hour_sin", "hour_cos", "is_weekend"]


class ConsumptionClassifier:
    """Build features, train two models and compare them on held-out data."""

    def __init__(self, hourly_kwh, train_fraction=0.8, random_state=42):
        if not 0.5 <= train_fraction < 0.95:
            raise ValueError("train_fraction must be between 0.5 and 0.95.")
        self._hourly = hourly_kwh.asfreq("h")   # complete hourly index; gaps become NaN
        self._train_fraction = train_fraction
        self._random_state = random_state
        self._results = {}
        self._info = {}

    @property
    def results(self):
        return self._results

    @property
    def info(self):
        return dict(self._info)

    def _build_dataset(self):
        """Create features X (from the past) and next-hour labels y, then split by time."""
        series = self._hourly
        cutoff_pos = int(len(series) * self._train_fraction)
        cutoff = series.index[cutoff_pos]                    # first hour of the test period

        train_hours = series[series.index < cutoff].dropna()
        threshold = float(train_hours.median())              # threshold from TRAINING data only

        labels = (series > threshold).astype(float).where(series.notna())   # NaN stays NaN
        feats = pd.DataFrame(index=series.index)
        feats["lag_1h"] = series
        feats["lag_2h"] = series.shift(1)
        feats["lag_3h"] = series.shift(2)
        feats["lag_24h"] = series.shift(23)                  # same hour yesterday as seen from t+1
        feats["rolling_mean_3h"] = series.rolling(3).mean()  # uses hours t-2..t only
        next_hour = series.index + pd.Timedelta(hours=1)     # calendar features of the hour to predict
        feats["hour_sin"] = np.sin(2 * np.pi * next_hour.hour / 24)
        feats["hour_cos"] = np.cos(2 * np.pi * next_hour.hour / 24)
        feats["is_weekend"] = (next_hour.dayofweek >= 5).astype(float)
        target = labels.shift(-1)                            # label of hour t+1

        data = feats.assign(target=target).dropna()
        target_time = data.index + pd.Timedelta(hours=1)
        train = data[target_time < cutoff]
        test = data[data.index >= cutoff - pd.Timedelta(hours=1)]
        test = test[(test.index + pd.Timedelta(hours=1)) >= cutoff]

        if len(train) < 50 or len(test) < 20:
            raise ValueError("Not enough data for a train/test split (need more days of data).")
        self._info = {
            "threshold_kwh": round(threshold, 4),
            "threshold_rule": "median of training hours only",
            "split_time": str(cutoff),
            "train_rows": len(train),
            "test_rows": len(test),
            "train_high_share": round(float(train["target"].mean()), 3),
            "test_high_share": round(float(test["target"].mean()), 3),
        }
        return (train[FEATURE_NAMES], train["target"].astype(int),
                test[FEATURE_NAMES], test["target"].astype(int))

    @staticmethod
    def _metrics(y_true, y_pred):
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        }

    def run(self):
        """Train both models on the same split and evaluate on the unseen test rows."""
        x_train, y_train, x_test, y_test = self._build_dataset()

        models = {
            "Logistic Regression": make_pipeline(StandardScaler(),
                                                 LogisticRegression(max_iter=1000, random_state=self._random_state)),
            "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=10,
                                                    random_state=self._random_state),
        }
        self._results = {}
        for name, model in models.items():
            model.fit(x_train, y_train)
            result = self._metrics(y_test, model.predict(x_test))
            if name == "Decision Tree":
                importance = dict(zip(FEATURE_NAMES, model.feature_importances_))
                result["top_features"] = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:3]
            else:
                coefs = dict(zip(FEATURE_NAMES, model[-1].coef_[0]))
                result["top_features"] = sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
            self._results[name] = result

        # Reference point: "next hour has the same label as the current hour".
        current_label = (x_test["lag_1h"] > self._info["threshold_kwh"]).astype(int)
        self._results["Baseline (repeat current hour)"] = self._metrics(y_test, current_label)
        return self._results

    def summary_text(self):
        """Human-readable comparison table."""
        if not self._results:
            return "Models have not been run yet."
        i = self._info
        lines = [
            f"Threshold: {i['threshold_kwh']} kWh ({i['threshold_rule']})",
            f"Chronological split at {i['split_time']}: {i['train_rows']} train rows, {i['test_rows']} test rows",
            f"High-consumption share: train {i['train_high_share']:.0%}, test {i['test_high_share']:.0%}",
            "",
            f"{'Model':34s}{'Accuracy':>9s}{'Precision':>10s}{'Recall':>8s}{'F1':>7s}",
        ]
        for name, r in self._results.items():
            lines.append(f"{name:34s}{r['accuracy']:9.3f}{r['precision']:10.3f}{r['recall']:8.3f}{r['f1']:7.3f}")
        lines.append("")
        for name, r in self._results.items():
            if "top_features" in r:
                feats = ", ".join(f"{k} ({v:+.2f})" for k, v in r["top_features"])
                lines.append(f"{name} - most influential features: {feats}")
        return "\n".join(lines)
