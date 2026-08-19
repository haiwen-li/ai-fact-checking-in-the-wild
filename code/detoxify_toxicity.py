"""
Score note text for toxicity with Detoxify (unitary/unbiased-toxic-roberta).

Writes data/note_toxicity.csv, one row per note:
    noteId, toxicity, severe_toxicity, obscene, identity_attack, insult,
    threat, sexual_explicit, toxicity_model
`toxicity` is the overall P(toxic) and is the feature used downstream; the
other six ride along for description. `toxicity_model` records the scorer
("detoxify-unbiased").

URLs are stripped before scoring, matching the other text features in
note_text_style_features.py.
"""

import argparse
import os
import re
import pandas as pd

from detoxify import Detoxify

from process_data import load_analysis_notes
from report_utils import DATA_DIR

OUT_PATH = os.path.join(DATA_DIR, "note_toxicity.csv")
URL_RE = re.compile(r"https?://[^\s]+")

CHECKPOINT = "unbiased"
MODEL_TAG = "detoxify-unbiased"
LABELS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "identity_attack",
    "insult",
    "threat",
    "sexual_explicit",
]

def _score_batches(model, texts: list[str], batch_size: int) -> pd.DataFrame:
    rows = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        result = model.predict(batch)
        for i in range(len(batch)):
            rows.append({label: float(result[label][i]) for label in LABELS})
        done = min(start + batch_size, len(texts))
        if done % (batch_size * 10) == 0 or done == len(texts):
            print(f"  {done:,}/{len(texts):,}")
    return pd.DataFrame(rows, columns=LABELS)


def main():
    ap = argparse.ArgumentParser(description="Toxicity scoring with Detoxify")
    ap.add_argument(
        "--device",
        default="cpu",
        help="cpu, cuda, or mps (Apple Silicon). Default cpu.",
    )
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="Score only N notes")
    args = ap.parse_args()

    print(f"Loading Detoxify('{CHECKPOINT}') on {args.device} ...")
    model = Detoxify(CHECKPOINT, device=args.device)

    notes = load_analysis_notes()[["noteId", "summary"]]
    if args.limit:
        notes = notes.head(args.limit)

    texts = [
        URL_RE.sub(" ", str(s) if pd.notna(s) else "").strip()
        for s in notes["summary"]
    ]
    n_empty = sum(1 for t in texts if not t)
    print(f"{len(texts):,} notes to score ({n_empty:,} have no text after URL removal).")
    # The model needs non-empty input; empty notes are scored on a single
    # space and their rows are blanked out afterwards.
    safe = [t if t else " " for t in texts]

    scores = _score_batches(model, safe, args.batch_size)

    out = pd.DataFrame({"noteId": notes["noteId"].values})
    for label in LABELS:
        out[label] = scores[label].values
    out["toxicity_model"] = MODEL_TAG

    empty_mask = [not t for t in texts]
    if any(empty_mask):
        out.loc[empty_mask, LABELS] = float("nan")

    if os.path.exists(OUT_PATH):
        try:
            prev = pd.read_csv(OUT_PATH)
            who = (
                prev["toxicity_model"].dropna().iloc[0]
                if "toxicity_model" in prev.columns and len(prev)
                else "unknown"
            )
            if who != MODEL_TAG:
                print(f"Note: overwriting scores previously written by '{who}'.")
        except Exception:  # noqa: BLE001
            pass

    out.to_csv(OUT_PATH, index=False)
    t = out["toxicity"].dropna()
    print(f"\nWrote {OUT_PATH}: {len(out):,} rows")
    print(
        f"  toxicity: mean {t.mean():.4f}, median {t.median():.4f}, "
        f"90th pct {t.quantile(0.9):.4f}, max {t.max():.4f}"
    )
    print(
        f"  above 0.5: {int((t > 0.5).sum()):,} notes "
        f"({(t > 0.5).mean() * 100:.2f}%)"
    )
    print(
        "\nNow re-run:\n"
        "  python analysis.py --analysis text\n"
        "  Rscript rating_analysis.R\n"
        "and the toxicity rows will appear in the report."
    )


if __name__ == "__main__":
    main()
