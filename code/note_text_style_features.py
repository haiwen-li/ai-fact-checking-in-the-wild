"""
Compute per-note text-style features (word count, URL count,
Flesch-Kincaid grade, VADER valence), used by the Note Characteristics
comparison (analysis.py::text_features_analysis) and by the SI "Textual
Features as Predictors of Ratings" mediation models (rating_analysis.R's
m1_style / m1_style_x).

Writes data/note_text_features.csv, one row per note:
    noteId, tweetId, writer, n_words, n_urls, flesch_kincaid_grade,
    vader_compound

URLs are stripped before any linguistic measure is computed. 
Toxicity is scored separately by detoxify_toxicity.py, since it needs a downloaded model.
"""

import os
import re

import numpy as np
import pandas as pd
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from process_data import load_analysis_notes
from report_utils import DATA_DIR

FEATURES_PATH = os.path.join(DATA_DIR, "note_text_features.csv")

URL_RE = re.compile(r"https?://[^\s]+")


def compute_text_features(notes: pd.DataFrame) -> pd.DataFrame:
    """
    Build the note-level text feature table.
    """
    vader = SentimentIntensityAnalyzer()
    rows = []
    for _, r in notes.iterrows():
        raw = str(r["summary"]) if pd.notna(r["summary"]) else ""
        urls = URL_RE.findall(raw)
        prose = URL_RE.sub(" ", raw).strip()
        n_words = len(prose.split())
        vs = (
            vader.polarity_scores(prose)
            if prose
            else {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 0.0}
        )
        grade = (
            float(textstat.flesch_kincaid_grade(prose)) if n_words >= 3 else np.nan
        )

        rows.append(
            {
                "noteId": r["noteId"],
                "tweetId": r["tweetId"],
                "writer": r["writer"],
                "n_words": n_words,
                "n_urls": len(urls),
                "flesch_kincaid_grade": grade,
                "vader_compound": vs["compound"],
            }
        )
    return pd.DataFrame(rows)


def main():
    notes = load_analysis_notes()
    feats = compute_text_features(notes)
    feats.to_csv(FEATURES_PATH, index=False)
    print(f"Saved text features to {FEATURES_PATH}")


if __name__ == "__main__":
    main()
