"""Text processing, Track A: custom cleaning/normalization, then TF-IDF (Lab 12).

Energy notes are stored one per line in a TXT file, e.g.
    2007-01-05 | Heater was left on all night, bill looks high
The date part is optional.
"""

import string
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.exceptions import DataValidationError
from src.utils import RAW_DIR

NOTES_FILE = RAW_DIR / "energy_notes.txt"

STOPWORDS = frozenset(
    "a an the and or but if so then of to in on at for with from by is are was were be been it its this that "
    "these those i we you they he she my our your their there here as not no very too just also again all "
    "any some after before over under up down out off again than can could would should will did do does".split()
)

# Small custom normalization map (synonyms / abbreviations -> one canonical word).
NORMALIZE = {
    "ac": "aircon", "a/c": "aircon", "airconditioner": "aircon", "conditioner": "aircon",
    "geyser": "heater", "boiler": "heater", "heating": "heater", "heaters": "heater",
    "fridge": "refrigerator", "freezer": "refrigerator",
    "tv": "television", "telly": "television",
    "bulbs": "light", "bulb": "light", "lights": "light", "lamp": "light", "lamps": "light",
    "bills": "bill", "kwh": "unit", "units": "unit",
}


def clean_text(text):
    """Lowercase, strip punctuation and digits, remove stopwords, normalize words.

    Returns:
        Cleaned string of space-separated tokens.
    """
    text = text.lower().replace("/", "")                       # "a/c" -> "ac"
    keep = string.ascii_lowercase + " "
    text = "".join(ch if ch in keep else " " for ch in text)  # drop punctuation, digits, symbols
    tokens = []
    for word in text.split():
        if word in STOPWORDS or len(word) < 2:
            continue
        word = NORMALIZE.get(word, word)
        word = _simple_stem(word)
        tokens.append(NORMALIZE.get(word, word))
    return " ".join(tokens)


NO_STEM = frozenset({"evening", "morning", "during", "used", "bedroom"})


def _simple_stem(word):
    """Very small rule-based suffix remover (plural/-ing/-ed), kept deliberately simple."""
    if word in NO_STEM:
        return word
    for suffix, minimum in (("ing", 5), ("ed", 5), ("ies", 5), ("es", 5), ("s", 4)):
        if word.endswith(suffix) and len(word) >= minimum and not word.endswith("ss"):
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


class NoteProcessor:
    """Load notes, clean them and build TF-IDF features for exploration."""

    def __init__(self, notes_path=None):
        self._path = Path(notes_path) if notes_path else NOTES_FILE
        self._raw, self._clean = [], []
        self._vectorizer, self._matrix = None, None

    @property
    def path(self):
        return self._path

    def load(self):
        """Read the notes file (TXT). Returns the number of notes loaded."""
        if not self._path.exists():
            raise DataValidationError(f"Notes file not found: {self._path}", self._path)
        try:
            with open(self._path, encoding="utf-8") as handle:
                lines = [line.strip() for line in handle if line.strip()]
        except (OSError, UnicodeDecodeError) as exc:
            raise DataValidationError(f"Could not read notes: {exc}", self._path) from exc

        self._raw = [line.split("|", 1)[1].strip() if "|" in line else line for line in lines]
        self._clean = [clean_text(note) for note in self._raw]
        keep = [i for i, c in enumerate(self._clean) if c]     # drop notes that became empty
        self._raw = [self._raw[i] for i in keep]
        self._clean = [self._clean[i] for i in keep]
        if len(self._clean) < 2:
            raise DataValidationError("Need at least 2 non-empty notes for TF-IDF.", self._path)

        self._vectorizer = TfidfVectorizer()
        self._matrix = self._vectorizer.fit_transform(self._clean)
        return len(self._clean)

    def add_note(self, note, date_text=None):
        """Append one note to the TXT file."""
        line = f"{date_text} | {note}" if date_text else note
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def top_terms(self, n=10):
        """Terms with the highest total TF-IDF weight across all notes: list of (term, weight)."""
        weights = self._matrix.sum(axis=0).A1
        terms = self._vectorizer.get_feature_names_out()
        return sorted(zip(terms, weights), key=lambda pair: pair[1], reverse=True)[:n]

    def similar_notes(self, query, n=3):
        """Find notes most similar to a query (cosine similarity on TF-IDF vectors)."""
        cleaned = clean_text(query)
        if not cleaned:
            return []
        scores = cosine_similarity(self._vectorizer.transform([cleaned]), self._matrix)[0]
        ranked = sorted(zip(scores, self._raw), key=lambda pair: pair[0], reverse=True)
        return [(float(s), note) for s, note in ranked[:n] if s > 0]

    def example_cleaning(self, count=3):
        """Return (original, cleaned) pairs to show what the cleaning does."""
        return list(zip(self._raw[:count], self._clean[:count]))

    @property
    def shape(self):
        return self._matrix.shape
