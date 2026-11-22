"""Rule-based energy-saving recommendations (Lab 05: ABC, inheritance, overriding, polymorphism).

All thresholds below are simple heuristics chosen for this prototype. The suggestions
are estimates, not guaranteed savings.
"""

from abc import ABC, abstractmethod


class EnergyRecommendationStrategy(ABC):
    """Abstract base class defining the recommendation interface."""

    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @abstractmethod
    def generate(self):
        """Return a list of recommendation strings."""

    def render(self):
        """Format the recommendations for printing (shared by all subclasses)."""
        lines = self.generate()
        body = "\n".join(f"  {i}. {text}" for i, text in enumerate(lines, start=1))
        return f"--- {self._name} ---\n{body}"


class UsageBasedRecommendation(EnergyRecommendationStrategy):
    """Recommendations derived from observed consumption patterns."""

    def __init__(self, stats, tariff_per_kwh):
        super().__init__("Usage-based recommendations")
        self._stats = stats
        self._tariff = tariff_per_kwh

    def generate(self):  # overrides the abstract method
        s, tips = self._stats, []
        tips.append(f"Your busiest hour is about {s['peak_hour']:02d}:00. "
                    "Avoid running several heavy appliances at the same time then.")

        base_per_day = s["base_load_kwh"] * 24
        if s["base_load_kwh"] > 0.35:
            monthly = base_per_day * 30 * self._tariff
            tips.append(f"Your lowest usage is about {s['base_load_kwh']:.2f} kWh per hour, which suggests "
                        f"always-on devices. Switching off standby devices could matter "
                        f"(base load alone is roughly {monthly:.2f} per 30 days at your tariff).")
        if s["evening_share"] > 0.30:
            tips.append(f"{s['evening_share']:.0%} of daily energy is used between 18:00 and 22:00. "
                        "Move flexible tasks (washing machine, dishwasher, ironing) to other hours, "
                        "especially if your tariff is cheaper off-peak.")
        if s["night_share"] > 0.20:
            tips.append(f"{s['night_share']:.0%} of daily energy is used between 00:00 and 06:00. "
                        "Check for devices left on overnight (heaters, water heater, chargers).")
        if s["weekend_ratio"] > 1.15:
            tips.append(f"Weekend days use about {s['weekend_ratio']:.2f}x the energy of weekdays. "
                        "Review what runs while people are home.")
        elif s["weekend_ratio"] < 0.85:
            tips.append(f"Weekdays use more energy than weekends (weekend/weekday = {s['weekend_ratio']:.2f}). "
                        "Check what stays on while the house is empty.")
        if len(tips) == 1:
            tips.append("No unusual patterns were found by the rules; usage looks balanced.")
        return tips


class ApplianceBasedRecommendation(EnergyRecommendationStrategy):
    """Recommendations from user-provided appliance information."""

    # Keyword -> advice. A dict is used for fast lookup by appliance-name keyword.
    ADVICE = {
        "heater": "Use a thermostat/timer and insulate the room; heating is usually the largest load.",
        "geyser": "Lower the water temperature and use a timer so it heats only when needed.",
        "water": "Lower the water temperature and use a timer so it heats only when needed.",
        "ac": "Set the thermostat 1-2 degrees higher and clean the filters regularly.",
        "air": "Set the thermostat 1-2 degrees higher and clean the filters regularly.",
        "dryer": "Air-dry clothes when possible, or run full loads only.",
        "iron": "Iron clothes in one batch instead of several short sessions.",
        "oven": "Avoid preheating for long and batch your cooking.",
        "washing": "Wash full loads at lower temperatures.",
    }

    def __init__(self, appliances, tariff_per_kwh):
        """appliances: list of dicts with keys name, watts, hours_per_day."""
        super().__init__("Appliance-based recommendations")
        self._appliances = appliances
        self._tariff = tariff_per_kwh

    def generate(self):  # overrides the abstract method
        if not self._appliances:
            return ["No appliance information provided. Add appliances in the menu to get tips."]

        rows = [(a["name"], a["watts"] * a["hours_per_day"] / 1000.0, a) for a in self._appliances]
        rows.sort(key=lambda r: r[1], reverse=True)  # biggest daily energy first
        total_kwh = sum(kwh for _, kwh, _ in rows)

        tips = []
        top_name, top_kwh, top = rows[0]
        share = top_kwh / total_kwh if total_kwh else 0
        monthly_cost = top_kwh * 30 * self._tariff
        tips.append(f"'{top_name}' uses the most energy: {top_kwh:.2f} kWh/day ({share:.0%} of listed "
                    f"appliances, about {monthly_cost:.2f} per 30 days).")
        saved = top["watts"] / 1000.0 * 30 * self._tariff
        tips.append(f"Using '{top_name}' one hour less per day would save roughly {saved:.2f} per 30 days.")

        for name, kwh, appliance in rows:
            words = set(name.lower().replace("-", " ").split())  # whole words only ("ac" must not match "machine")
            for keyword, advice in self.ADVICE.items():
                if keyword in words:
                    if kwh >= 0.5 or appliance["hours_per_day"] >= 4:
                        tips.append(f"{name}: {advice}")
                    break
        always_on = list(filter(lambda a: a["hours_per_day"] >= 24 and a["watts"] >= 100, self._appliances))
        for appliance in always_on:
            tips.append(f"{appliance['name']} runs 24 h/day at {appliance['watts']:.0f} W: "
                        "check that it is efficient and properly set.")
        return tips
