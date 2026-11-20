"""Small helper functions: validated console input and path constants (Labs 01-02, 04)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"


def ask_float(prompt, minimum=None, maximum=None, default=None):
    """Ask for a number until valid input is given.

    Args:
        prompt: Text shown to the user.
        minimum: Smallest accepted value (inclusive), or None.
        maximum: Largest accepted value (inclusive), or None.
        default: Value returned when the user just presses Enter, or None.

    Returns:
        The validated float.
    """
    while True:
        text = input(prompt).strip()
        if text == "" and default is not None:
            return float(default)
        try:
            value = float(text)  # type conversion
        except ValueError:
            print("  Invalid input: please enter a number.")
            continue
        if minimum is not None and value < minimum:
            print(f"  Value must be at least {minimum}.")
        elif maximum is not None and value > maximum:
            print(f"  Value must be at most {maximum}.")
        else:
            return value


def ask_choice(prompt, valid_choices):
    """Ask until the user enters one of the valid choices (case-insensitive)."""
    valid = {str(c).lower() for c in valid_choices}  # set: fast membership test
    while True:
        text = input(prompt).strip().lower()
        if text in valid:
            return text
        print(f"  Invalid choice. Options: {', '.join(sorted(valid))}")


def ask_text(prompt, allow_empty=False):
    """Ask for a text string; re-ask if empty and empty is not allowed."""
    while True:
        text = input(prompt).strip()
        if text or allow_empty:
            return text
        print("  Input cannot be empty.")


def pause():
    """Wait for Enter so output stays visible before the menu redraws."""
    input("\nPress Enter to return to the menu...")
