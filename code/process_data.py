"""
Data processing pipeline for the LLM vs. human Community Notes analysis.

Handles:
1. Identifying bot (LLM) and human notes from raw Community Notes data
2. Filtering ratings to relevant notes and raters
3. Building the common-rater ratings subset

The raw Community Notes snapshot (cndata02032026/) is not distributed with
this repository; replication starts from the pre-computed CSVs in data/
(fast-start mode, the default). --no-fast-start rebuilds those CSVs from the
raw snapshot and is only runnable if you download the snapshot yourself.

Usage
-----
# Fast-start: load pre-computed CSVs from data/ and print a summary
python process_data.py

# Full run: process from raw Community Notes TSVs
python process_data.py --no-fast-start

# Build complete_rater_ratings.csv (for the common-rater analysis)
python process_data.py --common-raters
"""

import argparse
import json
import os
import sys

import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(_SCRIPT_DIR, "data")

# Raw Community Notes data snapshot from 02/03/2026
# (download from https://communitynotes.x.com/guide/en/under-the-hood/download-data)
CN_DATA_DIR = os.path.join(_SCRIPT_DIR, "cndata02032026")

NOTES_TSV_PATH = os.path.join(CN_DATA_DIR, "notes-00000.tsv")
RATINGS_DIR = os.path.join(CN_DATA_DIR, "ratings")
USER_ENROLLMENT_PATH = os.path.join(CN_DATA_DIR, "userEnrollment-00000.tsv")
HELPFULNESS_SCORES_PATH = os.path.join(
    CN_DATA_DIR, "outputs", "helpfulness_scores.tsv"
)
SCORED_NOTES_PATH = os.path.join(CN_DATA_DIR, "scored_notes.tsv")
NOTE_STATUS_PATH = os.path.join(CN_DATA_DIR, "noteStatusHistory-00000.tsv")

# All API-retrieved tweet IDs (posts targeted by the LLM writer)
BOT_TWEET_IDS_PATH = os.path.join(DATA_DIR, "tids_api_retrieved.txt")

# Pre-processed CSVs (fast-start inputs)
ALL_NOTES_PATH = os.path.join(DATA_DIR, "all_notes.csv")
FILTERED_RATINGS_PATH = os.path.join(DATA_DIR, "filtered_ratings.csv")

# Study window start: Nov 1, 2025 ET, in epoch milliseconds. Notes created
# before this (and media notes) are excluded from all analyses.
STUDY_START_MILLIS = 1761969600000

# Bot account identifiers
with open(os.path.join(DATA_DIR, "api_account_ids.json"), "r") as f:
    API_ACCOUNT_IDS = json.load(f)


# =============================================================================
# 1. Data loading and preparation
# =============================================================================


def apply_note_filters(notes_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the analysis-sample filters: no media notes, study window only."""
    return notes_df[
        (notes_df["isMediaNote"] == 0)
        & (notes_df["createdAtMillis"] > STUDY_START_MILLIS)
    ]


def load_analysis_notes() -> pd.DataFrame:
    """Load data/all_notes.csv with the analysis-sample filters applied."""
    return apply_note_filters(pd.read_csv(ALL_NOTES_PATH))


def prepare_and_load_data(fast_start: bool = True):
    """
    Load and prepare the analysis dataset.

    Parameters
    ----------
    fast_start : bool
        If True, load pre-computed CSVs from DATA_DIR.
        If False, process raw Community Notes TSVs from scratch.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (notes_df, ratings_df)
    """
    if fast_start:
        return _load_precomputed()
    return _process_from_raw()


def _load_precomputed():
    """Load pre-computed CSVs (fast-start mode)."""
    print("=" * 80)
    print("FAST START: Loading pre-computed data from data/")
    print("=" * 80)

    if not os.path.exists(ALL_NOTES_PATH) or not os.path.exists(FILTERED_RATINGS_PATH):
        print("ERROR: Pre-computed files not found. Run with --no-fast-start first.")
        sys.exit(1)

    print(f"\nLoading notes from: {ALL_NOTES_PATH}")
    notes_df = load_analysis_notes()

    print(f"  Total notes: {len(notes_df):,}")
    print(f"  Bot notes: {(notes_df['writer'] == 'bot').sum():,}")
    print(f"  Human notes: {(notes_df['writer'] == 'human').sum():,}")
    print(f"  Unique tweets: {notes_df['tweetId'].nunique():,}")

    print(f"\nLoading filtered ratings from: {FILTERED_RATINGS_PATH}")
    ratings_df = pd.read_csv(FILTERED_RATINGS_PATH)
    ratings_df = ratings_df[ratings_df["noteId"].isin(notes_df["noteId"])]
    print(f"  Total ratings: {len(ratings_df):,}")
    print(f"  Unique notes: {ratings_df['noteId'].nunique():,}")
    print(f"  Unique raters: {ratings_df['raterParticipantId'].nunique():,}")

    return notes_df, ratings_df


def _process_from_raw():
    """Process raw Community Notes TSVs from scratch."""
    print("=" * 80)
    print("PREPARING DATA FROM RAW TSVs")
    print("=" * 80)

    # --- Step 1: Identify bot notes ---
    print("\nLoading notes data...")
    notes_df = pd.read_csv(
        NOTES_TSV_PATH,
        sep="\t",
        dtype={"tweetId": "Int64", "noteId": "Int64"},
        low_memory=False,
    )
    notes_df = notes_df[
        [
            "noteId",
            "noteAuthorParticipantId",
            "createdAtMillis",
            "tweetId",
            "summary",
            "isMediaNote",
            "classification",
        ]
    ]

    print("\nLoading bot tweet IDs...")
    bot_tids = set()
    with open(BOT_TWEET_IDS_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                bot_tids.add(int(line))

    bot_notes = notes_df[
        (notes_df["noteAuthorParticipantId"].isin(API_ACCOUNT_IDS.values()))
        & (notes_df["tweetId"].isin(bot_tids))
    ].copy()
    bot_notes["writer"] = "bot"

    print(f"\nBot notes: {len(bot_notes):,}")
    print(f"  Unique tweets: {bot_notes['tweetId'].nunique():,}")

    # --- Step 2: Identify human notes on the same tweets ---
    print("\nLoading user enrollment data...")
    user_enrollment = pd.read_csv(USER_ENROLLMENT_PATH, sep="\t", low_memory=False)
    api_ids = user_enrollment[user_enrollment["enrollmentState"] == "apiEarnedIn"][
        "participantId"
    ].unique()

    human_notes = notes_df[
        (~notes_df["noteAuthorParticipantId"].isin(api_ids))
        & (notes_df["tweetId"].isin(bot_notes["tweetId"].values))
    ].copy()
    human_notes["writer"] = "human"

    print(f"\nHuman notes: {len(human_notes):,}")
    print(f"  Unique tweets: {human_notes['tweetId'].nunique():,}")

    # --- Step 3: Combine with note scores and status, save ---
    all_notes = pd.concat([bot_notes, human_notes])
    print(
        f"\nCombined: {len(all_notes):,} notes, {all_notes['tweetId'].nunique():,} tweets"
    )

    scored_notes = pd.read_csv(
        SCORED_NOTES_PATH, sep="\t", dtype={"noteId": "Int64"}, low_memory=False
    )
    all_notes = all_notes.merge(
        scored_notes[
            [
                "noteId",
                "finalRatingStatus",
                "numRatings",
                "coreNoteIntercept",
                "coreNoteFactor1",
            ]
        ],
        on="noteId",
        how="left",
    )

    print("\nLoading note status history...")
    note_status_history = pd.read_csv(NOTE_STATUS_PATH, sep="\t", low_memory=False)
    all_notes = all_notes.merge(
        note_status_history[
            [
                "noteId",
                "timestampMillisOfCurrentStatus",
                "timestampMillisOfLatestNonNMRStatus",
            ]
        ],
        on="noteId",
        how="left",
    )

    all_notes.to_csv(ALL_NOTES_PATH, index=False)
    print(f"Saved notes to: {ALL_NOTES_PATH}")

    # --- Step 4: Load and filter ratings ---
    print("\n" + "=" * 80)
    print("LOADING RATINGS")
    print("=" * 80)

    helpfulness_df = pd.read_csv(HELPFULNESS_SCORES_PATH, sep="\t")
    helpfulness_df = helpfulness_df.dropna(
        subset=["coreRaterFactor1", "coreRaterIntercept"]
    )
    print(f"  Raters with valid factors: {len(helpfulness_df):,}")

    notes_ids = all_notes["noteId"].unique()
    raters_ids = helpfulness_df["raterParticipantId"].unique()

    ratings_files = [
        f
        for f in os.listdir(RATINGS_DIR)
        if f.endswith(".tsv") and f.startswith("ratings-")
    ]
    print(f"  Found {len(ratings_files)} rating files")

    all_ratings = []
    for file in ratings_files:
        file_path = os.path.join(RATINGS_DIR, file)
        print(f"  Processing {file}...")
        ratings_chunk = pd.read_csv(file_path, sep="\t", low_memory=False)
        ratings_chunk = ratings_chunk[
            (ratings_chunk["noteId"].isin(notes_ids))
            & (ratings_chunk["raterParticipantId"].isin(raters_ids))
        ]
        all_ratings.append(ratings_chunk)

    ratings_df = pd.concat(all_ratings, ignore_index=True)
    print(f"\nTotal ratings: {len(ratings_df):,}")

    ratings_df = ratings_df.merge(
        helpfulness_df[["raterParticipantId", "coreRaterFactor1", "coreRaterIntercept"]],
        on="raterParticipantId",
        how="left",
    )

    ratings_df.to_csv(FILTERED_RATINGS_PATH, index=False)
    print(f"Saved filtered ratings to: {FILTERED_RATINGS_PATH}")

    # --- Step 5: Rating-level analysis dataset (for the mixed models) ---
    notes_filtered = apply_note_filters(all_notes)
    filtered_note_ids = notes_filtered["noteId"].unique()

    ratings_w_rater_factors = ratings_df[
        ratings_df["noteId"].isin(filtered_note_ids)
    ].dropna(subset=["coreRaterFactor1"])
    print(f"Ratings with rater factors: {len(ratings_w_rater_factors):,}")

    notes_cols = ["noteId", "writer"]
    if "tweetId" not in ratings_w_rater_factors.columns:
        notes_cols.append("tweetId")
    ratings_analysis_df = ratings_w_rater_factors.merge(
        notes_filtered[notes_cols], on="noteId", how="inner"
    )

    ratings_analysis_df["AI"] = (ratings_analysis_df["writer"] == "bot").astype(int)
    ratings_analysis_df["abs_coreRaterFactor1"] = ratings_analysis_df[
        "coreRaterFactor1"
    ].abs()

    score_map = {"HELPFUL": 1.0, "SOMEWHAT_HELPFUL": 0.5, "NOT_HELPFUL": 0.0}
    ratings_analysis_df["rating_score"] = ratings_analysis_df["helpfulnessLevel"].map(
        score_map
    )
    ratings_analysis_df = ratings_analysis_df.dropna(subset=["rating_score"])

    out_path = os.path.join(DATA_DIR, "ratings_analysis_df.csv")
    ratings_analysis_df.to_csv(out_path, index=False)
    print(f"Saved analysis data to: {out_path}")

    return all_notes, ratings_df


# =============================================================================
# 2. Common-rater filtering
# =============================================================================


def filter_to_common_raters(
    ratings_df: pd.DataFrame,
    notes_df: pd.DataFrame,
    output_filename: str = "complete_rater_ratings.csv",
) -> pd.DataFrame:
    """
    For each tweet, identify "common raters" who rated ALL notes on that
    tweet. Restricts to tweets with at least one human note.

    Parameters
    ----------
    ratings_df : pd.DataFrame
        Ratings with columns noteId, raterParticipantId.
    notes_df : pd.DataFrame
        Notes with columns noteId, tweetId, writer.

    Returns
    -------
    pd.DataFrame
        Filtered ratings from common raters only.
    """
    # Restrict to tweets with at least one human note
    tweets_with_human = notes_df[notes_df["writer"] == "human"]["tweetId"].unique()
    notes_subset = notes_df[notes_df["tweetId"].isin(tweets_with_human)]

    # Merge ratings with notes to get tweetId
    ratings_with_tweet = ratings_df.merge(
        notes_subset[["noteId", "tweetId"]],
        on="noteId",
        how="inner",
    )

    # Per tweet: how many notes exist
    notes_per_tweet = (
        notes_subset.groupby("tweetId")["noteId"]
        .nunique()
        .reset_index()
        .rename(columns={"noteId": "n_notes_on_tweet"})
    )

    # Per (tweet, rater): how many notes the rater rated
    ratings_per_rater_tweet = (
        ratings_with_tweet.groupby(["tweetId", "raterParticipantId"])["noteId"]
        .nunique()
        .reset_index()
        .rename(columns={"noteId": "n_notes_rated"})
    )

    # Find common raters
    rater_tweet_stats = ratings_per_rater_tweet.merge(
        notes_per_tweet, on="tweetId", how="inner"
    )
    common_rater_tweets = rater_tweet_stats[
        rater_tweet_stats["n_notes_rated"] == rater_tweet_stats["n_notes_on_tweet"]
    ][["tweetId", "raterParticipantId"]]

    # Filter ratings
    ratings_filtered = ratings_with_tweet.merge(
        common_rater_tweets,
        on=["tweetId", "raterParticipantId"],
        how="inner",
    )

    # Save common-rater ratings (input for recomputing noteParams.tsv)
    common_rater_ratings_path = os.path.join(DATA_DIR, output_filename)
    ratings_filtered.to_csv(common_rater_ratings_path, index=False)
    print(f"\nSaved common-rater ratings to: {common_rater_ratings_path}")

    return ratings_filtered


def precompute_human_crh_hit_rate():
    """
    Per-author CRH/CRNH counts for ALL human Community Notes writers, used by
    the writer-percentile benchmarking in analysis.py (Appendix B). Requires
    the raw snapshot (cndata02032026/); the output, data/human_crh_hit_rate.csv,
    is distributed with the repository.
    """
    notes = pd.read_csv(
        NOTES_TSV_PATH,
        sep="\t",
        dtype={"noteId": "Int64", "tweetId": "Int64"},
        low_memory=False,
    )
    user_enrollment = pd.read_csv(USER_ENROLLMENT_PATH, sep="\t", low_memory=False)
    scored_notes = pd.read_csv(
        SCORED_NOTES_PATH, sep="\t", dtype={"noteId": "Int64"}, low_memory=False
    )
    notes = notes.merge(
        scored_notes[["noteId", "finalRatingStatus"]], on="noteId", how="left"
    )
    notes = notes.dropna(subset=["finalRatingStatus"])
    human_users = user_enrollment[
        user_enrollment["enrollmentState"] != "apiEarnedIn"
    ]["participantId"].unique()

    human_notes = notes[notes["noteAuthorParticipantId"].isin(human_users)]
    author_stats = (
        human_notes.groupby("noteAuthorParticipantId")
        .agg(
            n_total=("noteId", "count"),
            n_crh=(
                "finalRatingStatus",
                lambda s: (s == "CURRENTLY_RATED_HELPFUL").sum(),
            ),
            n_crnh=(
                "finalRatingStatus",
                lambda s: (s == "CURRENTLY_RATED_NOT_HELPFUL").sum(),
            ),
        )
        .reset_index()
        .rename(columns={"noteAuthorParticipantId": "participantId"})
    )

    out_path = os.path.join(DATA_DIR, "human_crh_hit_rate.csv")
    author_stats.to_csv(out_path, index=False)
    print(f"Saved human CRH hit rate to: {out_path}")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Data processing pipeline for the LLM vs. human notes analysis"
    )
    parser.add_argument(
        "--no-fast-start",
        action="store_true",
        help="Process from raw Community Notes TSVs (requires the raw snapshot)",
    )
    parser.add_argument(
        "--common-raters",
        action="store_true",
        help="Build complete_rater_ratings.csv",
    )
    parser.add_argument(
        "--human-crh-hit-rate",
        action="store_true",
        help="Rebuild data/human_crh_hit_rate.csv (requires the raw snapshot)",
    )
    args = parser.parse_args()

    notes_df, ratings_df = prepare_and_load_data(fast_start=not args.no_fast_start)

    if args.common_raters:
        print("\nFiltering to common raters...")
        ratings_filtered = filter_to_common_raters(ratings_df, notes_df)
        print(f"  Total ratings: {len(ratings_filtered):,}")
        print(f"  Unique notes: {ratings_filtered['noteId'].nunique():,}")
        print(f"  Unique raters: {ratings_filtered['raterParticipantId'].nunique():,}")

    if args.human_crh_hit_rate:
        precompute_human_crh_hit_rate()

    print("\nData loaded successfully.")
    print(f"  Notes: {len(notes_df):,}")
    print(f"  Ratings: {len(ratings_df):,}")


if __name__ == "__main__":
    main()
