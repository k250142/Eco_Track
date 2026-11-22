"""EcoTrack: Household Energy Consumption & Cost Analyzer (console application).

Run from the project folder:
    python main.py
"""

import sys

from src.data_manager import EnergyDataManager
from src.exceptions import EcoTrackError
from src.text_processor import NoteProcessor
from src.utils import ask_choice, ask_float, ask_text, pause

MENU = """
==================== ECO TRACK ====================
 1. Load and validate energy data
 2. Consumption analysis (hourly / daily / monthly)
 3. Cost estimation
 4. NumPy day-by-hour analysis
 5. Generate charts
 6. Machine learning: Logistic Regression vs Decision Tree
 7. Text processing of energy notes (TF-IDF)
 8. Energy-saving recommendations
 9. Weather information (Open-Meteo)
10. Exit
===================================================="""


class EcoTrackApp:
    """Holds the application state and implements each menu action."""

    def __init__(self):
        self.manager = EnergyDataManager()
        self.notes = NoteProcessor()

    # ---- helpers ---------------------------------------------------------------
    def analyzer(self):
        """Create the analyzer on demand; raises DataValidationError if data is not loaded."""
        if self._analyzer is None:
            self._analyzer = EnergyAnalyzer(self.manager.hourly)
        return self._analyzer

    # ---- menu actions ----------------------------------------------------------
    def load_data(self):
        report = self.manager.load_raw()
        print("\nData loaded and validated:")
        print(f"  File:              {report['raw_file']}")
        print(f"  Period:            {report['date_from']}  to  {report['date_to']}")
        print(f"  Rows read:         {report['rows_read']}")
        print(f"  Valid rows:        {report['valid_rows']}")
        print(f"  Missing ('?') rows:{report['rows_with_missing_values']:>6}")
        print(f"  Malformed rows:    {report['malformed_rows']}")
        print(f"  Hours kept:        {report['hours_kept']} (dropped for low coverage: "
              f"{report['hours_dropped_low_coverage']})")
        for example in report["malformed_examples"][:3]:
            print(f"    skipped -> {example[:110]}")

    def text_processing(self):
        while True:
            choice = ask_choice("\n [a] Analyse notes  [b] Add a note  [c] Search notes  [m] Back: ",
                                ["a", "b", "c", "m"])
            if choice == "m":
                return
            if choice == "b":
                note = ask_text("Note text: ")
                date_text = ask_text("Date (YYYY-MM-DD, optional): ", allow_empty=True)
                self.notes.add_note(note, date_text or None)
                print("Note saved.")
                continue
            count = self.notes.load()
            if choice == "a":
                print(f"\n{count} notes, TF-IDF matrix shape {self.notes.shape}.")
                print("Cleaning examples (original -> cleaned):")
                for original, cleaned in self.notes.example_cleaning(3):
                    print(f"  {original}\n    -> {cleaned}")
                print("Top terms by TF-IDF weight:")
                for term, weight in self.notes.top_terms(10):
                    print(f"  {term:15s}{weight:.2f}")
            else:
                query = ask_text("Search words: ")
                matches = self.notes.similar_notes(query)
                if not matches:
                    print("No similar notes found.")
                for score, note in matches:
                    print(f"  [{score:.2f}] {note}")

def main():
    app = EcoTrackApp()
    actions = {
        "1": app.load_data,
        "7": app.text_processing,
    }
    while True:
        print(MENU)
        choice = input("Choose 1-10: ").strip()
        if choice == "10":
            print("Goodbye!")
            return
        action = actions.get(choice)
        if action is None:
            if choice.isdigit() and 1 <= int(choice) <= 9:
                print("This option is not available in this version yet.")
            else:
                print("Invalid choice. Please enter a number from 1 to 10.")
            continue
        try:
            action()
        except EcoTrackError as exc:
            print(f"\nError: {exc}")
        except ValueError as exc:
            print(f"\nInvalid input: {exc}")
        except KeyboardInterrupt:
            print("\nCancelled.")
        except EOFError:
            raise
        except Exception as exc:  # last-resort guard so the menu never crashes
            print(f"\nUnexpected error ({type(exc).__name__}): {exc}")
        pause()


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        sys.exit(0)
