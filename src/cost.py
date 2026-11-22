"""CostCalculator: electricity cost estimates from kWh and a user tariff (Lab 04, 05)."""


class CostCalculator:
    """Estimate electricity costs from energy (kWh) and a price per kWh."""

    DEFAULT_TARIFF = 0.15  # currency units per kWh; the user should enter their own

    def __init__(self, tariff_per_kwh=DEFAULT_TARIFF, currency="units"):
        self._currency = currency
        self._tariff = None
        self.tariff = tariff_per_kwh  # goes through the validating setter

    @property
    def tariff(self):
        return self._tariff

    @tariff.setter
    def tariff(self, value):
        if value is None or value <= 0:
            raise ValueError("Tariff must be a positive number (cost per kWh).")
        self._tariff = float(value)

    @property
    def currency(self):
        return self._currency

    def cost_for_energy(self, kwh):
        """Cost of a given amount of energy in kWh."""
        return kwh * self._tariff

    def total_cost(self, hourly_kwh):
        """Total cost and total energy of a series of hourly kWh values."""
        total_kwh = float(hourly_kwh.sum())
        return total_kwh, self.cost_for_energy(total_kwh)

    def daily_costs(self, hourly_kwh):
        """Cost per day (Series), counting only days with all 24 hours measured."""
        resampled = hourly_kwh.resample("D")
        totals, counts = resampled.sum(), resampled.count()
        return (totals[counts == 24] * self._tariff).rename("daily_cost")

    def monthly_costs(self, hourly_kwh):
        """Measured cost per calendar month (Series) and hours measured (Series)."""
        resampled = hourly_kwh.resample("MS")
        return (resampled.sum() * self._tariff).rename("measured_cost"), resampled.count()

    def projected_monthly_bill(self, mean_daily_kwh, days=30):
        """Estimated bill for `days` days at the average daily energy use."""
        return self.cost_for_energy(mean_daily_kwh * days)

    def appliance_daily_cost(self, watts, hours_per_day):
        """Daily energy (kWh) and cost for an appliance. kWh = W x hours / 1000."""
        kwh = watts * hours_per_day / 1000.0
        return kwh, self.cost_for_energy(kwh)
