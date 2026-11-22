"""WeatherService: Open-Meteo API via Requests with status checks and offline JSON cache (Lab 06).

Input methods (documented in the README):
  * a place name, e.g. "Karachi"  -> resolved to coordinates by the Open-Meteo geocoding API
  * coordinates as "latitude,longitude", e.g. "24.86,67.01"

Weather is shown as context only; it is not evidence that weather caused past consumption.
"""

import json
from datetime import datetime

import requests

from src.exceptions import WeatherServiceError
from src.utils import PROCESSED_DIR

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_SECONDS = 10


class WeatherService:
    """Retrieve, validate and cache weather information."""

    def __init__(self, cache_path=None, timeout=TIMEOUT_SECONDS):
        self._cache_path = cache_path or (PROCESSED_DIR / "weather_cache.json")
        self._timeout = timeout

    # ---- location handling ---------------------------------------------------
    @staticmethod
    def parse_coordinates(text):
        """Return (lat, lon) if text looks like 'lat,lon', otherwise None.

        Raises:
            ValueError: the text has two numbers but they are outside valid ranges.
        """
        parts = [p.strip() for p in text.split(",")]
        if len(parts) != 2:
            return None
        try:
            lat, lon = float(parts[0]), float(parts[1])
        except ValueError:
            return None
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Latitude must be -90..90 and longitude -180..180.")
        return lat, lon

    def _get_json(self, url, params):
        """GET a URL, check the HTTP status code and decode JSON."""
        try:
            response = requests.get(url, params=params, timeout=self._timeout)
        except requests.exceptions.RequestException as exc:   # no network, timeout, DNS ...
            raise ConnectionError(f"network unavailable ({type(exc).__name__})") from exc
        if response.status_code != 200:
            raise ConnectionError(f"service returned HTTP {response.status_code}")
        try:
            return response.json()
        except ValueError as exc:
            raise ConnectionError("service returned invalid JSON") from exc

    def geocode(self, place):
        """Resolve a place name to (name, country, lat, lon) using the geocoding API."""
        data = self._get_json(GEOCODE_URL, {"name": place, "count": 1, "language": "en", "format": "json"})
        results = data.get("results") or []
        if not results:
            raise WeatherServiceError(f"Place not found: '{place}'. Try another spelling or 'lat,lon'.")
        top = results[0]
        try:
            return top["name"], top.get("country", ""), float(top["latitude"]), float(top["longitude"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WeatherServiceError("Geocoding response was missing coordinates.") from exc

    # ---- forecast + cache ----------------------------------------------------
    def get_weather(self, location_text):
        """Return a weather dict for a place name or 'lat,lon'.

        Online: fetch, validate, save to the cache. Offline or service error: use the
        cached copy if one exists, otherwise raise WeatherServiceError.

        Returns:
            dict with keys: label, current, daily, source ('live' or 'cache'), fetched_at.
        """
        location_text = location_text.strip()
        if not location_text:
            raise ValueError("Location cannot be empty.")
        coords = self.parse_coordinates(location_text)   # may raise ValueError
        key = location_text.lower() if coords is None else f"{coords[0]:.2f},{coords[1]:.2f}"

        try:
            if coords is None:
                name, country, lat, lon = self.geocode(location_text)
                label = f"{name}, {country}".strip(", ")
            else:
                lat, lon = coords
                label = f"{lat:.2f}, {lon:.2f}"
            payload = self._get_json(FORECAST_URL, {
                "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": 3,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            })
            weather = self._validate_and_shape(payload, label)
            weather["source"] = "live"
            self._save_cache(key, weather)
            return weather
        except (ConnectionError, WeatherServiceError) as exc:
            cached = self._load_cache(key)
            if cached is not None:
                cached["source"] = "cache"
                cached["note"] = f"Showing saved data because live data failed: {exc}"
                return cached
            raise WeatherServiceError(
                f"Could not get weather ({exc}) and no cached data exists for '{location_text}'."
            ) from exc

    @staticmethod
    def _validate_and_shape(payload, label):
        """Check that the response has the expected fields and keep only what is needed."""
        try:
            current = payload["current"]
            daily = payload["daily"]
            shaped = {
                "label": label,
                "fetched_at": datetime.now().isoformat(timespec="seconds"),
                "current": {
                    "temperature_c": float(current["temperature_2m"]),
                    "humidity_pct": float(current["relative_humidity_2m"]),
                    "wind_kmh": float(current["wind_speed_10m"]),
                },
                "daily": [
                    {"date": d, "max_c": float(hi), "min_c": float(lo), "rain_mm": float(rain or 0)}
                    for d, hi, lo, rain in zip(daily["time"], daily["temperature_2m_max"],
                                               daily["temperature_2m_min"], daily["precipitation_sum"])
                ],
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise WeatherServiceError(f"Unexpected weather response format: {exc}") from exc
        if not shaped["daily"]:
            raise WeatherServiceError("Weather response contained no daily data.")
        return shaped

    def _load_all_cache(self):
        try:
            with open(self._cache_path, encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _load_cache(self, key):
        return self._load_all_cache().get(key)

    def _save_cache(self, key, weather):
        cache = self._load_all_cache()
        cache[key] = weather
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "w", encoding="utf-8") as handle:
                json.dump(cache, handle, indent=2)
        except OSError:
            pass  # failing to cache must not break the live result

    @staticmethod
    def format_weather(weather):
        """Printable text for a weather dict."""
        cur = weather["current"]
        lines = [f"Weather for {weather['label']} (source: {weather['source']}, fetched {weather['fetched_at']})"]
        if weather.get("note"):
            lines.append(f"  NOTE: {weather['note']}")
        lines.append(f"  Now: {cur['temperature_c']:.1f} C, humidity {cur['humidity_pct']:.0f}%, "
                     f"wind {cur['wind_kmh']:.0f} km/h")
        for day in weather["daily"]:
            lines.append(f"  {day['date']}: {day['min_c']:.1f} to {day['max_c']:.1f} C, rain {day['rain_mm']:.1f} mm")
        lines.append("  (Weather is context only; it does not prove why past usage changed.)")
        return "\n".join(lines)
