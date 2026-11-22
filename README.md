# EcoTrack: Household Energy Consumption & Cost Analyzer

FAST-NUCES, AL-2001 Programming for Artificial Intelligence (Lab), Fall 2026, Section BAI-3A

Group: Syed Abdullah Shah (25K-0142), Bhawani Shanker (25K-0044)

Menu-driven Python console application that analyzes household electricity consumption.
**This is work in progress (development version 8 of 10).** The full README will be completed in the final version.

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Choose menu option 1 first to load the data.

## Menu options available so far

| # | Option |
|---|---|
| 1 | Load and validate data |
| 2 | Consumption analysis |
| 3 | Cost estimation |
| 4 | NumPy day-by-hour analysis |
| 5 | Charts |
| 6 | Machine learning |
| 7 | Text processing (TF-IDF) |
| 8 | Recommendations |

Still to be added: Weather (Open-Meteo).

## Data

UCI Individual Household Electric Power Consumption dataset:
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
The file in `data/raw/` is a small subset (see `src/prepare_data.py`).

## AI assistance

The initial code was generated with AI assistance (Claude, by Anthropic) and then reviewed and modified by the group.
