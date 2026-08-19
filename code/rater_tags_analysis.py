"""
Rater-provided tags on LLM vs human notes (paper Section 3.5, table
`tab:rater_tags_mechanism`).

When submitting a rating, raters can select tags describing why a note is
helpful or unhelpful. Following the platform's eligibility rules:
- helpful* tags are analyzed among HELPFUL or SOMEWHAT_HELPFUL ratings
- notHelpful* tags among NOT_HELPFUL or SOMEWHAT_HELPFUL ratings
(SOMEWHAT_HELPFUL ratings are eligible for both tag families.)

For each tag we report three estimands:
- rating-weighted: share of eligible ratings selecting the tag (pools all
  ratings, so heavily rated notes weigh more)
- equal-note: tag share computed per note, then averaged across notes
- matched-tweet: restrict to tweets with >=1 AI and >=1 human note, average
  note-level tag shares within tweet and writer type, and test the paired
  AI-minus-human difference across tweets (paired t-test; this is the
  inferential estimand reported in the paper)

Results are written to the 'rater-tags' section of outputs/analysis_report.md.

Usage: python rater_tags_analysis.py
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

from process_data import load_analysis_notes
from report_utils import DATA_DIR, update_report_section

_REPORT: list[str] = []


def _r(msg: str = ""):
    print(msg)
    _REPORT.append(msg)


def _fmt_pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.1f}"


def _fmt_p(x: float) -> str:
    return "" if pd.isna(x) else f"{x:.2g}"


def _matched_tweets(notes: pd.DataFrame) -> set:
    """Tweets with at least one bot and at least one human note."""
    by_tweet_writer = notes.groupby(["tweetId", "writer"]).size().unstack(fill_value=0)
    return set(
        by_tweet_writer[
            (by_tweet_writer.get("bot", 0) > 0) & (by_tweet_writer.get("human", 0) > 0)
        ].index
    )


def _tag_rows_for_family(
    ratings: pd.DataFrame,
    tags: list[str],
    eligible_levels: list[str],
    matched_tweet_ids: set,
) -> list[dict]:
    """
    Compute the rating-weighted, equal-note, and matched-tweet estimands for
    each tag in one family. Returns one dict per tag with keys
    'tag', 'rating', 'note', 'matched', sorted by descending combined
    rating-weighted prevalence.
    """
    eligible = ratings[ratings["helpfulnessLevel"].isin(eligible_levels)].copy()
    for tag in tags:
        eligible[tag] = pd.to_numeric(eligible[tag], errors="coerce").fillna(0)

    rating_ns = eligible.groupby("writer").size()
    rows = {}

    for tag in tags:
        tag_counts = eligible.groupby("writer")[tag].sum()
        if tag_counts.sum() == 0:
            continue

        # Rating-weighted: share of eligible ratings carrying the tag
        rating_row = {
            "ai_pct": tag_counts.get("bot", np.nan) / rating_ns.get("bot", np.nan),
            "hu_pct": tag_counts.get("human", np.nan) / rating_ns.get("human", np.nan),
        }

        # Equal-note: per-note tag share, averaged across notes
        per_note = eligible.groupby(
            ["tweetId", "noteId", "writer"], as_index=False
        ).agg(tag_count=(tag, "sum"), n_eligible=(tag, "size"))
        per_note["tag_prop"] = per_note["tag_count"] / per_note["n_eligible"]
        note_means = per_note.groupby("writer")["tag_prop"].mean()
        note_row = {
            "ai_pct": note_means.get("bot", np.nan),
            "hu_pct": note_means.get("human", np.nan),
        }

        # Matched-tweet: average note-level shares within (tweet, writer),
        # then paired t-test on the per-tweet AI-minus-human differences
        matched_per_note = per_note[per_note["tweetId"].isin(matched_tweet_ids)]
        tweet_writer = (
            matched_per_note.groupby(["tweetId", "writer"], as_index=False)["tag_prop"]
            .mean()
            .pivot(index="tweetId", columns="writer", values="tag_prop")
        )
        if {"bot", "human"}.issubset(tweet_writer.columns):
            paired = tweet_writer.dropna(subset=["bot", "human"]).copy()
            paired["diff"] = paired["bot"] - paired["human"]
        else:
            paired = pd.DataFrame(columns=["bot", "human", "diff"])

        if len(paired) >= 2:
            diff = paired["diff"]
            if diff.std(ddof=1) > 0:
                t_stat, p = stats.ttest_1samp(diff, 0, nan_policy="omit")
            else:
                t_stat, p = (
                    (0.0, 1.0) if np.isclose(diff.mean(), 0) else (np.nan, np.nan)
                )
        else:
            t_stat, p = np.nan, np.nan
        matched_row = {
            "ai_pct": paired["bot"].mean() if len(paired) else np.nan,
            "hu_pct": paired["human"].mean() if len(paired) else np.nan,
            "diff_pct": paired["diff"].mean() if len(paired) else np.nan,
            "n_tweets": len(paired),
            "t": t_stat,
            "p": p,
        }

        rows[tag] = {
            "tag": tag,
            "rating": rating_row,
            "note": note_row,
            "matched": matched_row,
        }

    def prevalence(row):
        r = row["rating"]
        return (0 if pd.isna(r["ai_pct"]) else r["ai_pct"]) + (
            0 if pd.isna(r["hu_pct"]) else r["hu_pct"]
        )

    return sorted(rows.values(), key=lambda row: -prevalence(row))


def tag_analysis(notes: pd.DataFrame, ratings: pd.DataFrame):
    helpful_tags = [
        c
        for c in ratings.columns
        if c.startswith("helpful") and c not in ("helpful", "helpfulnessLevel")
    ]
    nothelpful_tags = [
        c for c in ratings.columns if c.startswith("notHelpful") and c != "notHelpful"
    ]

    ratings = ratings.merge(
        notes[["noteId", "writer", "tweetId"]], on="noteId", how="inner"
    )
    matched_tweet_ids = _matched_tweets(notes)

    _r(
        "SOMEWHAT_HELPFUL ratings are eligible for both helpful and not-helpful tags. "
        "Rating-weighted rows are descriptive only; inference is from paired "
        "tweet-level AI-minus-human differences after averaging note-level tag "
        "proportions within tweet and writer type."
    )
    _r(
        f"Matched-tweet estimand restricted to {len(matched_tweet_ids):,} tweets "
        "with both AI and human notes."
    )

    for eligible_levels, tags, label in [
        (["HELPFUL", "SOMEWHAT_HELPFUL"], helpful_tags, "helpful"),
        (["NOT_HELPFUL", "SOMEWHAT_HELPFUL"], nothelpful_tags, "not-helpful"),
    ]:
        sub = ratings[ratings["helpfulnessLevel"].isin(eligible_levels)]
        n_ai = (sub["writer"] == "bot").sum()
        n_hu = (sub["writer"] == "human").sum()
        rows = _tag_rows_for_family(ratings, tags, eligible_levels, matched_tweet_ids)

        _r(
            f"\n### Tags on {label} or somewhat-helpful ratings "
            f"(AI: {n_ai:,} eligible ratings, human: {n_hu:,})"
        )
        _r("")
        _r(
            "| Tag | Rating-weighted AI % | Rating-weighted human % | Equal-note AI % "
            "| Equal-note human % | Matched-tweet AI % | Matched-tweet human % "
            "| AI-human pp | Paired-tweet t | p | Paired tweets |"
        )
        _r("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            rating = row["rating"]
            note = row["note"]
            matched = row["matched"]
            _r(
                f"| {row['tag']} | {_fmt_pct(rating['ai_pct'])} | {_fmt_pct(rating['hu_pct'])} "
                f"| {_fmt_pct(note['ai_pct'])} | {_fmt_pct(note['hu_pct'])} "
                f"| {_fmt_pct(matched['ai_pct'])} | {_fmt_pct(matched['hu_pct'])} "
                f"| {_fmt_pct(matched['diff_pct'])} | {matched['t']:.2f} "
                f"| {_fmt_p(matched['p'])} | {matched['n_tweets']:,} |"
            )


def main():
    notes = load_analysis_notes()
    ratings = pd.read_csv(os.path.join(DATA_DIR, "filtered_ratings.csv"))
    ratings = ratings[ratings["noteId"].isin(notes["noteId"])]

    tag_analysis(notes, ratings)

    path = update_report_section("rater-tags", _REPORT)
    print(f"\n[section 'rater-tags' updated in {path}]")


if __name__ == "__main__":
    main()
