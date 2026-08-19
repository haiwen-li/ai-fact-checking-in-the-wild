"""
Main analysis script.

Replicates the paper results that do not come from the R mixed models or
from a dedicated script (MBFC: source_quality_analysis.py; rater tags:
rater_tags_analysis.py; topic distributions: topic_distribution.py):

Results are written into the shared report, outputs/analysis_report.md
(see report_utils.py); each analysis updates its own section.

Usage
-----
# Run all default analyses
python analysis.py

# Run specific analyses
python analysis.py --analysis rating note timing_matched win_rate text pairwise_bt

# Both-note-posts replication
python analysis.py --both-notes-only

# common-rater analyses 
python analysis.py --analyze-with-common-raters

# Rater distribution comparison
python analysis.py --rater-distribution
"""

import argparse
import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

from process_data import (
    filter_to_common_raters,
    load_analysis_notes,
    prepare_and_load_data,
)
from report_utils import DATA_DIR, OUTPUT_DIR, update_report_section


# =============================================================================
# Report buffer
# =============================================================================

_REPORT_LINES: list[str] = []


def _report(msg: str = ""):
    """Print to console and append to report buffer."""
    print(msg)
    _REPORT_LINES.append(msg)


def _flush_section(section_id: str):
    """Write the buffered report lines into their section and clear the buffer."""
    global _REPORT_LINES
    path = update_report_section(section_id, _REPORT_LINES)
    _REPORT_LINES = []
    print(f"\n[section '{section_id}' updated in {path}]")


# =============================================================================
# Statistical helpers
# =============================================================================


def _bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction."""
    valid = ~np.isnan(pvals)
    if valid.sum() == 0:
        return pvals.copy()
    adj = np.full_like(pvals, np.nan)
    _, adj_valid, _, _ = multipletests(pvals[valid], method="fdr_bh")
    adj[valid] = adj_valid
    return adj


def _two_sample_tests(df: pd.DataFrame, col: str, label: str) -> dict:
    """Welch's t-test and Mann-Whitney U for bot vs human on a given column."""
    bot_vals = df[df["writer"] == "bot"][col].dropna()
    human_vals = df[df["writer"] == "human"][col].dropna()
    res = {
        "metric": label,
        "n_bot": len(bot_vals),
        "n_human": len(human_vals),
        "mean_bot": bot_vals.mean() if len(bot_vals) else np.nan,
        "mean_human": human_vals.mean() if len(human_vals) else np.nan,
        "median_bot": bot_vals.median() if len(bot_vals) else np.nan,
        "median_human": human_vals.median() if len(human_vals) else np.nan,
    }
    if len(bot_vals) > 1 and len(human_vals) > 1:
        t_stat, t_p = stats.ttest_ind(bot_vals, human_vals, equal_var=False)
        try:
            u_stat, u_p = stats.mannwhitneyu(
                bot_vals, human_vals, alternative="two-sided"
            )
        except ValueError:
            u_stat, u_p = np.nan, np.nan
    else:
        t_stat = t_p = u_stat = u_p = np.nan
    res.update({"t_stat": t_stat, "t_p": t_p, "u_stat": u_stat, "u_p": u_p})
    return res


def _report_two_sample(
    res: dict,
    fmt: str = ".2f",
    p_adj: float | None = None,
    primary_test: str = "t",
    skip_t: bool = False,
):
    """Write two-sample test results to the report."""
    _report(f"  n_bot={res['n_bot']}, n_human={res['n_human']}")
    _report(f"  mean: Bot={res['mean_bot']:{fmt}}  Human={res['mean_human']:{fmt}}")
    _report(
        f"  median: Bot={res['median_bot']:{fmt}}  Human={res['median_human']:{fmt}}"
    )
    if not skip_t and "t_p" in res and not (isinstance(res["t_p"], float) and np.isnan(res["t_p"])):
        if primary_test == "t":
            if p_adj is not None and not np.isnan(p_adj):
                _report(
                    f"  Welch t-test: t={res['t_stat']:.4f}, p={res['t_p']:.6f} (unadjusted), "
                    f"p_adj={p_adj:.6f} (BH)"
                )
            else:
                _report(f"  Welch t-test: t={res['t_stat']:.4f}, p={res['t_p']:.6f}")
        else:
            _report(f"  Welch t-test: t={res['t_stat']:.4f}, p={res['t_p']:.6f}")
    if "u_stat" in res and not (isinstance(res["u_p"], float) and np.isnan(res["u_p"])):
        if primary_test == "u":
            if p_adj is not None and not np.isnan(p_adj):
                _report(
                    f"  Mann-Whitney U: U={res['u_stat']:.2f}, p={res['u_p']:.6f} (unadjusted), "
                    f"p_adj={p_adj:.6f} (BH)"
                )
            else:
                _report(
                    f"  Mann-Whitney U: U={res['u_stat']:.2f}, p={res['u_p']:.6f}"
                )
        elif not skip_t:
            _report(f"  Mann-Whitney U: U={res['u_stat']:.2f}, p={res['u_p']:.6f}")


# =============================================================================
# LMM helpers
# =============================================================================


def _run_status_binary_lmm(
    notes_df: pd.DataFrame,
    status_col: str,
    status_val: str,
    context_label: str,
) -> dict | None:
    """Run LMM: (status==status_val) ~ AI + (1|tweetId)."""
    df = notes_df.copy()
    df["tweetId"] = df["tweetId"].astype(str)
    df = df.dropna(subset=[status_col, "writer"])
    df["y_binary"] = (df[status_col] == status_val).astype(int)
    df["AI"] = (df["writer"] == "bot").astype(int)

    if len(df) < 50:
        return None

    try:
        model = smf.mixedlm("y_binary ~ AI", data=df, groups=df["tweetId"])
        result = model.fit(reml=True)
        param_names = (
            list(result.model.exog_names) if hasattr(result.model, "exog_names") else []
        )
        idx = param_names.index("AI") if "AI" in param_names else -1
        if idx >= 0:
            return {
                "coef": float(result.params[idx]),
                "se": float(result.bse[idx]),
                "z": (
                    float(result.params[idx]) / float(result.bse[idx])
                    if result.bse[idx]
                    else np.nan
                ),
                "p": float(result.pvalues[idx]),
                "n": len(df),
            }
    except Exception:
        pass
    return None


def _report_lmm_result(
    label: str,
    res: dict | None,
    context_label: str,
    p_adj: float | None = None,
) -> None:
    """Report LMM result to the report buffer."""
    if res is None:
        _report(f"  {label} LMM ({context_label}): skipped or failed")
        return
    coef, se, z, p, n = res["coef"], res["se"], res["z"], res["p"], res["n"]
    if p_adj is not None and not np.isnan(p_adj):
        _report(
            f"  {label} LMM ({context_label}): AI coef={coef:.4f}, SE={se:.4f}, "
            f"z={z:.4f}, p={p:.6f} (unadjusted), p_adj={p_adj:.6f} (BH) (n={n:,})"
        )
    else:
        _report(
            f"  {label} LMM ({context_label}): AI coef={coef:.4f}, SE={se:.4f}, "
            f"z={z:.4f}, p={p:.6f} (n={n:,})"
        )


def _run_note_intercept_lmm(
    notes_df: pd.DataFrame,
    intercept_col: str,
    context_label: str,
) -> dict | None:
    """Run LMM: intercept ~ AI + (1|tweetId)."""
    df = notes_df.copy()
    df["tweetId"] = df["tweetId"].astype(str)
    df = df.dropna(subset=[intercept_col, "writer"])
    df["AI"] = (df["writer"] == "bot").astype(int)

    if len(df) < 50:
        return None

    try:
        model = smf.mixedlm(f"{intercept_col} ~ AI", data=df, groups=df["tweetId"])
        result = model.fit(reml=True)
        param_names = (
            list(result.model.exog_names) if hasattr(result.model, "exog_names") else []
        )
        idx = param_names.index("AI") if "AI" in param_names else -1
        if idx >= 0:
            coef = float(result.params[idx])
            se = float(result.bse[idx])
            return {
                "coef": coef,
                "se": se,
                "z": coef / se if se else np.nan,
                "p": float(result.pvalues[idx]),
                "n": len(df),
            }
    except Exception:
        pass
    return None



# =============================================================================
# Platform timing & exposure comparison
# =============================================================================


def human_bot_timing_analysis(notes_df: pd.DataFrame):
    """
    Compare submission timing between LLM and human notes.

    For tweets with both bot and human notes, compute what percentage of
    human notes were created before vs after the first bot note.
    """
    _report("### Human vs LLM note timing (createdAtMillis)")
    _report("")

    notes_with_ts = notes_df.dropna(subset=["createdAtMillis"]).copy()
    bot_notes = notes_with_ts[notes_with_ts["writer"] == "bot"]
    human_notes = notes_with_ts[notes_with_ts["writer"] == "human"]

    tweets_with_both = set(bot_notes["tweetId"].unique()) & set(
        human_notes["tweetId"].unique()
    )
    _report(f"Tweets with both human and bot notes: {len(tweets_with_both):,}")

    first_bot_ts = bot_notes.groupby("tweetId")["createdAtMillis"].min()

    human_before = 0
    human_after = 0
    hours_diffs = []

    for _, human_row in human_notes.iterrows():
        tid = human_row["tweetId"]
        if tid not in tweets_with_both:
            continue
        human_ts = human_row["createdAtMillis"]
        first_bot = first_bot_ts.loc[tid]
        hours_diff = (human_ts - first_bot) / 3.6e6
        hours_diffs.append(hours_diff)
        if human_ts < first_bot:
            human_before += 1
        else:
            human_after += 1

    total_human = human_before + human_after
    if total_human > 0:
        pct_before = human_before / total_human * 100
        pct_after = human_after / total_human * 100
        _report(f"\nHuman notes on tweets with both types: {total_human:,}")
        _report(
            f"  Created before first bot note: {human_before:,} ({pct_before:.1f}%)"
        )
        _report(
            f"  Created at or after first bot note: {human_after:,} ({pct_after:.1f}%)"
        )

        hours_arr = np.array(hours_diffs)
        _report("\n### Hours earlier/later (human vs first bot note)")
        _report(
            f"  Mean: {hours_arr.mean():.2f} h  (median: {np.median(hours_arr):.2f} h)"
        )
        _report(f"  Std: {hours_arr.std():.2f} h")
        _report(f"  Min: {hours_arr.min():.2f} h  Max: {hours_arr.max():.2f} h")
        _report(
            f"  Percentiles: 10th={np.percentile(hours_arr, 10):.2f} h, "
            f"25th={np.percentile(hours_arr, 25):.2f} h, "
            f"75th={np.percentile(hours_arr, 75):.2f} h, "
            f"90th={np.percentile(hours_arr, 90):.2f} h"
        )
    else:
        _report("\nNo human notes on tweets with both human and bot notes.")

    _report("")


# =============================================================================
# Rating-level analysis and ideology bucket means
# =============================================================================


def _assign_rater_bucket(factor: float) -> str:
    if factor < -0.15:
        return "left"
    elif factor > 0.15:
        return "right"
    else:
        return "neutral"


def _note_bucket_stats(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-note x rater-ideology-bucket rating counts and helpfulness stats.
    Expects ratings_df to already have "writer", "rater_bucket",
    "helpfulnessLevel", and "rating_score" columns.
    """
    note_bucket_stats = []
    for note_id, note_group in ratings_df.groupby("noteId"):
        writer = note_group["writer"].iloc[0]
        for bucket in ["left", "neutral", "right"]:
            bucket_ratings = note_group[note_group["rater_bucket"] == bucket]
            if len(bucket_ratings) == 0:
                continue
            total = len(bucket_ratings)
            helpful_count = (bucket_ratings["helpfulnessLevel"] == "HELPFUL").sum()
            not_helpful_count = (
                bucket_ratings["helpfulnessLevel"] == "NOT_HELPFUL"
            ).sum()
            note_bucket_stats.append(
                {
                    "noteId": note_id,
                    "writer": writer,
                    "rater_bucket": bucket,
                    "total_ratings": total,
                    "helpful_count": helpful_count,
                    "not_helpful_count": not_helpful_count,
                    "pct_helpful": helpful_count / total * 100.0,
                    "pct_not_helpful": not_helpful_count / total * 100.0,
                    "mean_score": bucket_ratings["rating_score"].mean(),
                }
            )
    return pd.DataFrame(note_bucket_stats)


def rating_counts_by_bucket_analysis(note_ids, context_label: str):
    """
    Compare the number of ratings received from each rater-ideology bucket
    (left/neutral/right) between AI and human notes, within a given note
    subset (e.g. an exposure-matched window from timing_matched_analysis).
    """
    _report(f"\n#### Ratings received by rater ideology bucket ({context_label})")
    _report("")

    ratings_df = pd.read_csv(os.path.join(DATA_DIR, "ratings_analysis_df.csv"))
    ratings_df = ratings_df[ratings_df["noteId"].astype(str).isin(note_ids)].copy()
    if ratings_df.empty:
        _report("  (no ratings in this subset)")
        _report("")
        return

    ratings_df["rater_bucket"] = ratings_df["coreRaterFactor1"].apply(
        _assign_rater_bucket
    )
    note_bucket_df = _note_bucket_stats(ratings_df)
    if note_bucket_df.empty:
        _report("  (no bucketed ratings)")
        _report("")
        return

    for bucket in ["left", "neutral", "right"]:
        bucket_data = note_bucket_df[note_bucket_df["rater_bucket"] == bucket]
        bot_counts = bucket_data[bucket_data["writer"] == "bot"]["total_ratings"]
        human_counts = bucket_data[bucket_data["writer"] == "human"]["total_ratings"]
        _report(
            f"  {bucket.capitalize()}: n_bot_notes={len(bot_counts)}, "
            f"n_human_notes={len(human_counts)}"
        )
        if len(bot_counts) == 0 or len(human_counts) == 0:
            _report("    (insufficient data)")
            continue
        _report(
            f"    ratings/note: mean Bot={bot_counts.mean():.2f}  "
            f"Human={human_counts.mean():.2f}  |  "
            f"median Bot={bot_counts.median():.1f}  Human={human_counts.median():.1f}"
        )
        if len(bot_counts) > 1 and len(human_counts) > 1:
            try:
                u_stat, u_p = stats.mannwhitneyu(
                    bot_counts, human_counts, alternative="two-sided"
                )
                _report(f"    Mann-Whitney U: U={u_stat:.2f}, p={u_p:.6f}")
            except ValueError:
                pass
    _report("")


def rating_analysis_by_bucket(
    both_writers_only: bool = False,
    save_figure: bool = True,
):
    """
    Compute per-note and aggregated % helpful / % not helpful by
    writer x rater_bucket, with 95% confidence intervals.

    If both_writers_only is True, restrict to tweets that have BOTH at least
    one LLM (bot) note and at least one human note.
    """
    _report(
        "### Ratings by rater ideology bucket"
        + (" (both-note posts)" if both_writers_only else "")
    )
    _report("")

    ratings_df = pd.read_csv(os.path.join(DATA_DIR, "ratings_analysis_df.csv"))
    print(f"Ratings analysis df: {ratings_df.shape}")

    if both_writers_only:
        notes_all = load_analysis_notes()
        bot_tweets = set(notes_all[notes_all["writer"] == "bot"]["tweetId"].astype(str))
        human_tweets = set(
            notes_all[notes_all["writer"] == "human"]["tweetId"].astype(str)
        )
        both_tweets = bot_tweets & human_tweets
        n_before = len(ratings_df)
        ratings_df = ratings_df[
            ratings_df["tweetId"].astype(str).isin(both_tweets)
        ].copy()
        _report(
            f"Restricted to tweets with both LLM and human notes: "
            f"{len(both_tweets):,} tweets, "
            f"{len(ratings_df):,}/{n_before:,} ratings retained"
        )
        _report("")
        print(f"Ratings analysis df (both writers only): {ratings_df.shape}")

    ratings_df["rater_bucket"] = ratings_df["coreRaterFactor1"].apply(
        _assign_rater_bucket
    )

    note_bucket_df = _note_bucket_stats(ratings_df)
    if note_bucket_df.empty:
        _report("  No per-note bucket stats.")
        return note_bucket_df, pd.DataFrame()

    # Aggregate across notes
    writer_bucket_summary = (
        note_bucket_df.groupby(["writer", "rater_bucket"])
        .agg(
            num_notes=("noteId", "count"),
            total_ratings=("total_ratings", "sum"),
            mean_pct_helpful=("pct_helpful", "mean"),
            std_pct_helpful=("pct_helpful", "std"),
            mean_pct_not_helpful=("pct_not_helpful", "mean"),
            std_pct_not_helpful=("pct_not_helpful", "std"),
        )
        .reset_index()
    )

    def _add_ci(row, mean_col, std_col):
        mean_val = row[mean_col]
        std_val = row[std_col]
        n = row["num_notes"]
        if n <= 1 or pd.isna(std_val):
            return pd.Series({"ci_lower": mean_val, "ci_upper": mean_val, "se": np.nan})
        se = std_val / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        return pd.Series(
            {
                "ci_lower": mean_val - t_crit * se,
                "ci_upper": mean_val + t_crit * se,
                "se": se,
            }
        )

    ci_h = writer_bucket_summary.apply(
        lambda r: _add_ci(r, "mean_pct_helpful", "std_pct_helpful"), axis=1
    )
    writer_bucket_summary[["ci_lower_helpful", "ci_upper_helpful", "se_helpful"]] = ci_h

    ci_nh = writer_bucket_summary.apply(
        lambda r: _add_ci(r, "mean_pct_not_helpful", "std_pct_not_helpful"), axis=1
    )
    writer_bucket_summary[
        ["ci_lower_not_helpful", "ci_upper_not_helpful", "se_not_helpful"]
    ] = ci_nh

    # Report
    _report("Mean % Helpful by writer and rater bucket (95% CI across notes):")
    for bucket in ["left", "neutral", "right"]:
        bucket_data = writer_bucket_summary[
            writer_bucket_summary["rater_bucket"] == bucket
        ]
        if bucket_data.empty:
            continue
        _report(f"\nBucket = {bucket}")
        for _, row in bucket_data.iterrows():
            ci_str = f"[{row['ci_lower_helpful']:.2f}, {row['ci_upper_helpful']:.2f}]"
            _report(
                f"  {row['writer']:>6}: mean={row['mean_pct_helpful']:6.2f}%  "
                f"(n={int(row['num_notes'])}, CI={ci_str})"
            )

    _report("Mean % Unhelpful by writer and rater bucket (95% CI across notes):")
    for bucket in ["left", "neutral", "right"]:
        bucket_data = writer_bucket_summary[
            writer_bucket_summary["rater_bucket"] == bucket
        ]
        if bucket_data.empty:
            continue
        _report(f"\nBucket = {bucket}")
        for _, row in bucket_data.iterrows():
            ci_str = f"[{row['ci_lower_not_helpful']:.2f}, {row['ci_upper_not_helpful']:.2f}]"
            _report(
                f"  {row['writer']:>6}: mean={row['mean_pct_not_helpful']:6.2f}%  "
                f"(n={int(row['num_notes'])}, CI={ci_str})"
            )

    # Save grouped bar chart (paper Figure 1)
    if not save_figure:
        _report("")
        return note_bucket_df, writer_bucket_summary

    bar_buckets = ["left", "neutral", "right"]
    bar_bucket_labels = ["Left-leaning", "Neutral", "Right-leaning"]
    bar_width = 0.18
    pair_gap = 0.06
    group_spacing = 1.5
    group_centers = [i * group_spacing for i in range(3)]

    bar_offsets = [
        -1.5 * bar_width - pair_gap / 2,
        -0.5 * bar_width - pair_gap / 2,
        0.5 * bar_width + pair_gap / 2,
        1.5 * bar_width + pair_gap / 2,
    ]
    bar_colors = ["#27ae60", "#a9dfbf", "#e67e22", "#f5cba7"]
    bar_labels = [
        "Human %Helpful",
        "LLM %Helpful",
        "Human %Not Helpful",
        "LLM %Not Helpful",
    ]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for g_idx, (bucket, gc) in enumerate(zip(bar_buckets, group_centers)):
        bucket_data = writer_bucket_summary[
            writer_bucket_summary["rater_bucket"] == bucket
        ]
        human_row = bucket_data[bucket_data["writer"] == "human"]
        bot_row = bucket_data[bucket_data["writer"] == "bot"]

        bar_values = [
            human_row["mean_pct_helpful"].values[0] if len(human_row) else 0,
            bot_row["mean_pct_helpful"].values[0] if len(bot_row) else 0,
            human_row["mean_pct_not_helpful"].values[0] if len(human_row) else 0,
            bot_row["mean_pct_not_helpful"].values[0] if len(bot_row) else 0,
        ]
        bar_errors = [
            (
                [
                    bar_values[0] - human_row["ci_lower_helpful"].values[0],
                    human_row["ci_upper_helpful"].values[0] - bar_values[0],
                ]
                if len(human_row)
                else [0, 0]
            ),
            (
                [
                    bar_values[1] - bot_row["ci_lower_helpful"].values[0],
                    bot_row["ci_upper_helpful"].values[0] - bar_values[1],
                ]
                if len(bot_row)
                else [0, 0]
            ),
            (
                [
                    bar_values[2] - human_row["ci_lower_not_helpful"].values[0],
                    human_row["ci_upper_not_helpful"].values[0] - bar_values[2],
                ]
                if len(human_row)
                else [0, 0]
            ),
            (
                [
                    bar_values[3] - bot_row["ci_lower_not_helpful"].values[0],
                    bot_row["ci_upper_not_helpful"].values[0] - bar_values[3],
                ]
                if len(bot_row)
                else [0, 0]
            ),
        ]

        for i, (val, err, color, label) in enumerate(
            zip(bar_values, bar_errors, bar_colors, bar_labels)
        ):
            ax.bar(
                gc + bar_offsets[i],
                val,
                width=bar_width,
                color=color,
                yerr=[[err[0]], [err[1]]],
                capsize=4,
                error_kw={"elinewidth": 1.5},
                label=label if g_idx == 0 else "_nolegend_",
            )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(bar_bucket_labels, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage (%)", fontsize=13)
    ax.set_title(
        "LLM vs Human: Mean % Helpful/Not Helpful by Rater Ideology",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "rating_analysis_llm_vs_human_barchart.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    _report(f"\nSaved bar chart: {fig_path}")
    _report("")

    return note_bucket_df, writer_bucket_summary


# =============================================================================
# Full-sample note-level outcomes
# =============================================================================


def note_level_analysis(notes_df: pd.DataFrame, plot_suffix: str = ""):
    """
    Compare bot vs human notes on note-level outcomes:
    - finalRatingStatus distribution and %CRH / %CRNH
    - coreNoteIntercept (helpfulness score)
    - numRatings
    - Time to CRH (hours) among CRH notes
    """
    
    _report(
        "### Note-level analysis: LLM vs human"
        + (f" ({plot_suffix})" if plot_suffix else "")
    )
    _report("")

    results = []

    # Status distribution and %CRH / %CRNH
    _report("Final rating status distribution by writer:")
    status_cross = pd.crosstab(
        notes_df["writer"], notes_df["finalRatingStatus"], margins=True
    )
    _report("```")
    _report(status_cross.to_string())
    _report("```")

    status_results = []
    for status_name, status_val in [
        ("%CRH", "CURRENTLY_RATED_HELPFUL"),
        ("%CRNH", "CURRENTLY_RATED_NOT_HELPFUL"),
    ]:
        bot_notes = notes_df[notes_df["writer"] == "bot"]
        human_notes = notes_df[notes_df["writer"] == "human"]
        bot_rate = (
            (bot_notes["finalRatingStatus"] == status_val).mean() * 100
            if len(bot_notes)
            else np.nan
        )
        human_rate = (
            (human_notes["finalRatingStatus"] == status_val).mean() * 100
            if len(human_notes)
            else np.nan
        )
        n_bot_total = len(bot_notes)
        n_human_total = len(human_notes)
        status_results.append(
            {
                "status_name": status_name,
                "bot_rate": bot_rate,
                "human_rate": human_rate,
            }
        )
        results.append(
            {
                "metric": status_name,
                "n_bot": n_bot_total,
                "n_human": n_human_total,
                "mean_bot": bot_rate,
                "mean_human": human_rate,
            }
        )

    # coreNoteIntercept
    res_help = _two_sample_tests(notes_df, "coreNoteIntercept", "coreNoteIntercept")
    results.append(res_help)

    # numRatings
    res_ratings = _two_sample_tests(notes_df, "numRatings", "numRatings")
    results.append(res_ratings)

    # Time to CRH

    crh_notes = notes_df[notes_df["finalRatingStatus"] == "CURRENTLY_RATED_HELPFUL"]
    crh_notes = crh_notes[
        crh_notes["timestampMillisOfLatestNonNMRStatus"].notna()
        & crh_notes["createdAtMillis"].notna()
    ]
    crh_notes["time_to_crh_hours"] = (
        crh_notes["timestampMillisOfLatestNonNMRStatus"] - crh_notes["createdAtMillis"]
    ) / 3.6e6

    res_time = _two_sample_tests(crh_notes, "time_to_crh_hours", "time_to_crh_hours")
    results.append(res_time)

    # Report proportion rates (descriptive only, no hypothesis tests)
    for sr in status_results:
        _report(
            f"  {sr['status_name']}: Bot={sr['bot_rate']:.2f}%  Human={sr['human_rate']:.2f}%"
        )

    # LMMs (unadjusted p-values; BH correction only in common-rater analysis)
    context_label = plot_suffix if plot_suffix else "Full sample"
    res_crh = _run_status_binary_lmm(
        notes_df, "finalRatingStatus", "CURRENTLY_RATED_HELPFUL", context_label
    )
    res_crnh = _run_status_binary_lmm(
        notes_df, "finalRatingStatus", "CURRENTLY_RATED_NOT_HELPFUL", context_label
    )
    res_help_lmm = _run_note_intercept_lmm(
        notes_df, "coreNoteIntercept", context_label
    )
    _report_lmm_result("CRH", res_crh, context_label)
    _report_lmm_result("CRNH", res_crnh, context_label)

    # coreNoteIntercept (descriptive only + LMM)
    _report("\n--- coreNoteIntercept ---")
    _report_two_sample(res_help, skip_t=True)
    _report_lmm_result("Note intercept", res_help_lmm, context_label)

    # numRatings (U-test only, no t-test)
    _report("\n--- numRatings ---")
    _report_two_sample(res_ratings, primary_test="u", skip_t=True)

    # Time to CRH (U-test only, no t-test)
    _report("\n--- Time to CRH (hours) among CRH notes ---")
    _report_two_sample(res_time, primary_test="u", skip_t=True)

    _report("")


# =============================================================================
# Sensitivity to minimum-rating thresholds
# =============================================================================

THRESHOLDS = [10, 30, 50]


def thresholds_analysis(notes_df: pd.DataFrame):
    """
    Note-level comparison (CRH rate, CRNH rate, helpfulness score, numRatings,
    time to CRH) restricted to notes with at least 10 / 30 / 50 ratings.
    """
    _report("### Sensitivity to minimum rating thresholds")
    _report("")
    for t in THRESHOLDS:
        subset = notes_df[notes_df["numRatings"] >= t]
        n_bot = (subset["writer"] == "bot").sum()
        n_human = (subset["writer"] == "human").sum()
        _report(f"\n#### numRatings >= {t}")
        _report(
            f"Notes retained: {len(subset):,} "
            f"(bot: {n_bot:,}/{(notes_df['writer'] == 'bot').sum():,}, "
            f"human: {n_human:,}/{(notes_df['writer'] == 'human').sum():,})"
        )
        if len(subset) == 0:
            continue

        for status_name, status_val in [
            ("%CRH", "CURRENTLY_RATED_HELPFUL"),
            ("%CRNH", "CURRENTLY_RATED_NOT_HELPFUL"),
        ]:
            bot_n = subset[subset["writer"] == "bot"]
            human_n = subset[subset["writer"] == "human"]
            bot_rate = (
                (bot_n["finalRatingStatus"] == status_val).mean() * 100
                if len(bot_n)
                else np.nan
            )
            human_rate = (
                (human_n["finalRatingStatus"] == status_val).mean() * 100
                if len(human_n)
                else np.nan
            )
            _report(f"  {status_name}: Bot={bot_rate:.2f}%  Human={human_rate:.2f}%")

        label = f"numRatings>={t}"
        res_crh = _run_status_binary_lmm(
            subset, "finalRatingStatus", "CURRENTLY_RATED_HELPFUL", label
        )
        res_crnh = _run_status_binary_lmm(
            subset, "finalRatingStatus", "CURRENTLY_RATED_NOT_HELPFUL", label
        )
        res_help = _run_note_intercept_lmm(subset, "coreNoteIntercept", label)
        _report_lmm_result("CRH", res_crh, label, None)
        _report_lmm_result("CRNH", res_crnh, label, None)

        _report("\n--- coreNoteIntercept ---")
        res = _two_sample_tests(subset, "coreNoteIntercept", "coreNoteIntercept")
        _report_two_sample(res, skip_t=True)
        _report_lmm_result("Note intercept", res_help, label, None)

        _report("\n--- numRatings ---")
        res = _two_sample_tests(subset, "numRatings", "numRatings")
        _report_two_sample(res, primary_test="u", skip_t=True)

        crh_notes = subset[subset["finalRatingStatus"] == "CURRENTLY_RATED_HELPFUL"]
        crh_notes = crh_notes[
            crh_notes["timestampMillisOfLatestNonNMRStatus"].notna()
            & crh_notes["createdAtMillis"].notna()
        ].copy()
        crh_notes["time_to_crh_hours"] = (
            crh_notes["timestampMillisOfLatestNonNMRStatus"]
            - crh_notes["createdAtMillis"]
        ) / 3.6e6
        _report("\n--- Time to CRH (hours) among CRH notes ---")
        res = _two_sample_tests(crh_notes, "time_to_crh_hours", "time_to_crh_hours")
        _report_two_sample(res, primary_test="u", skip_t=True)

    _report("")


# =============================================================================
# Timing-matched robustness
# =============================================================================


def timing_matched_analysis(notes_df: pd.DataFrame):
    """
    For each bot note, find human notes on the same tweet within
    +/- 30 / 60 / 90 minutes. Run note_level_analysis on matched subsets.
    """
    _report("### Creation-time-matched subsets")
    _report("")

    notes_with_ts = notes_df.dropna(subset=["createdAtMillis"]).copy()
    bot_notes = notes_with_ts[notes_with_ts["writer"] == "bot"]
    human_notes = notes_with_ts[notes_with_ts["writer"] == "human"]

    def _create_pairs(window_minutes: int):
        window_ms = window_minutes * 60 * 1000
        pairs = []
        matched_bot_ids = set()
        for _, bot_row in bot_notes.iterrows():
            tid = bot_row["tweetId"]
            t_bot = bot_row["createdAtMillis"]
            humans_on_post = human_notes[human_notes["tweetId"] == tid]
            for _, h_row in humans_on_post.iterrows():
                if abs(h_row["createdAtMillis"] - t_bot) <= window_ms:
                    pairs.append(
                        {
                            "tweetId": tid,
                            "bot_noteId": bot_row["noteId"],
                            "human_noteId": h_row["noteId"],
                        }
                    )
                    matched_bot_ids.add(bot_row["noteId"])
        n_bot = len(bot_notes)
        n_matched = len(matched_bot_ids)
        match_rate = n_matched / n_bot * 100.0 if n_bot > 0 else np.nan
        return pd.DataFrame(pairs), n_bot, n_matched, match_rate

    for w in [30, 60, 90]:
        pairs_df, n_bot, n_matched, rate = _create_pairs(w)
        _report(
            f"Window +/-{w:>3} min: matched {n_matched}/{n_bot} bot notes ({rate:.1f}%)"
        )

        if not pairs_df.empty:
            subset_ids = pd.unique(
                np.concatenate(
                    [
                        pairs_df["bot_noteId"].to_numpy(),
                        pairs_df["human_noteId"].to_numpy(),
                    ]
                )
            )
            subset_notes = notes_df[notes_df["noteId"].isin(subset_ids)].copy()
            note_level_analysis(subset_notes, plot_suffix=f"timing_{w}min")
            rating_counts_by_bucket_analysis(
                set(subset_ids.astype(str)), context_label=f"+/-{w} min window"
            )

    _report("")


# =============================================================================
# Within-post pairwise comparison of note helpfulness scores
# =============================================================================


def _binom_ci(k: int, n: int) -> tuple:
    """Wilson score interval."""
    if n == 0:
        return (np.nan, np.nan)
    z = 1.959963985
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (centre - half, centre + half)


def _build_matched_pairs(notes: pd.DataFrame) -> pd.DataFrame:
    """Enumerate every (LLM note, human note) pair on the same post."""
    n = notes.dropna(subset=["createdAtMillis"]).copy()
    bot = n[n["writer"] == "bot"]
    hum = n[n["writer"] == "human"]
    cols = ["noteId", "createdAtMillis", "coreNoteIntercept"]
    return bot[["tweetId"] + cols].merge(
        hum[["tweetId"] + cols], on="tweetId", suffixes=("_bot", "_human")
    )


def win_rate_analysis(notes_df: pd.DataFrame):
    """
    Appendix: on posts carrying both an LLM and a human note, how often does
    the LLM note end up with the higher platform helpfulness score
    (coreNoteIntercept)? Reports both a pair-level win rate (every LLM-human
    combination on a post counts once) and a post-weighted win rate (each
    pair weighted by 1/(pairs on that post), so every post counts once).
    """
    _report(
        "A 'win' means the LLM note has the higher `coreNoteIntercept`. Notes "
        "the scorer never assigned an intercept to (too few ratings) are "
        "excluded."
    )
    _report("")

    pairs_all = _build_matched_pairs(notes_df)
    usable = pairs_all.dropna(
        subset=["coreNoteIntercept_bot", "coreNoteIntercept_human"]
    )
    diff = usable["coreNoteIntercept_bot"] - usable["coreNoteIntercept_human"]
    wins = int((diff > 0).sum())
    n = len(usable)
    lo, hi = _binom_ci(wins, n)

    _report("### Pair-level win rate")
    _report("")
    _report(
        f"Scored pairs: {n:,} on {usable['tweetId'].nunique():,} posts "
        f"({len(pairs_all) - n:,} pairs dropped for missing scores)."
    )
    _report(
        f"LLM win rate: {wins / n * 100:.1f}% (95% CI [{lo * 100:.1f}, "
        f"{hi * 100:.1f}]). Mean score gap (LLM minus human): {diff.mean():+.3f}."
    )

    # Post-weighted: each pair weighted 1/(pairs on that post), so every post
    # contributes a total weight of one.
    u = usable.copy()
    u["win"] = (u["coreNoteIntercept_bot"] > u["coreNoteIntercept_human"]).astype(float)
    u["gap"] = u["coreNoteIntercept_bot"] - u["coreNoteIntercept_human"]
    per_post = u.groupby("tweetId").agg(win_share=("win", "mean"), gap=("gap", "mean"))
    n2 = len(per_post)
    win_rate = per_post["win_share"].mean() * 100 if n2 else np.nan
    sd = per_post["win_share"].std()
    half = (
        stats.t.ppf(0.975, df=n2 - 1) * sd / np.sqrt(n2)
        if n2 > 1 and sd == sd
        else np.nan
    )

    _report("")
    _report("### Post-weighted win rate")
    _report("")
    _report(
        f"Posts: {n2:,}. Post-weighted LLM win rate: {win_rate:.1f}% "
        f"(95% CI [{win_rate - half * 100:.1f}, {win_rate + half * 100:.1f}]). "
        f"Weighted mean score gap: {per_post['gap'].mean():+.3f}."
    )
    _report("")


# =============================================================================
# Writing and source analysis
# =============================================================================


def text_features_analysis(notes_df: pd.DataFrame):
    """
    Compare text features between bot and human notes:
    - Note length (word count): t-test
    - URL count: t-test
    - Top 10 cited domains by writer
    """
    _report("### Text features")
    _report("")

    def _extract_features(text):
        if pd.isna(text):
            return {"word_len": 0, "url_count": 0}
        s = str(text)
        return {
            "word_len": len(s.split()),
            "url_count": len(re.findall(r"https?://[^\s]+", s)),
        }

    feat_df = notes_df["summary"].apply(lambda s: pd.Series(_extract_features(s)))
    notes_with_features = notes_df.copy()
    for col in feat_df.columns:
        notes_with_features[col] = feat_df[col].values

    def _two_sample(col, label):
        bot_vals = notes_with_features[notes_with_features["writer"] == "bot"][
            col
        ].dropna()
        human_vals = notes_with_features[notes_with_features["writer"] == "human"][
            col
        ].dropna()
        res = {
            "metric": label,
            "n_bot": len(bot_vals),
            "n_human": len(human_vals),
            "mean_bot": bot_vals.mean() if len(bot_vals) else np.nan,
            "mean_human": human_vals.mean() if len(human_vals) else np.nan,
            "median_bot": bot_vals.median() if len(bot_vals) else np.nan,
            "median_human": human_vals.median() if len(human_vals) else np.nan,
        }
        if len(bot_vals) > 1 and len(human_vals) > 1:
            t_stat, t_p = stats.ttest_ind(bot_vals, human_vals, equal_var=False)
        else:
            t_stat = t_p = np.nan
        res.update({"t_stat": t_stat, "t_p": t_p})
        return res

    tests = [_two_sample("word_len", "word_len"), _two_sample("url_count", "url_count")]

    _report("#### Note length (word count)")
    _report_two_sample(tests[0], fmt=".1f")
    _report("")
    _report("#### Number of URLs")
    _report_two_sample(tests[1])
    _report("")

    # Readability, valence, toxicity (computed on note text with URLs
    # removed, by note_text_style_features.py and detoxify_toxicity.py).
    # Read from the precomputed CSVs (fast-start default); skip gracefully
    # if they have not been generated.
    style_path = os.path.join(DATA_DIR, "note_text_features.csv")
    tox_path = os.path.join(DATA_DIR, "note_toxicity.csv")
    if os.path.exists(style_path):
        style = pd.read_csv(style_path)[
            ["noteId", "flesch_kincaid_grade", "vader_compound"]
        ]
        style["noteId"] = style["noteId"].astype(notes_with_features["noteId"].dtype)
        notes_with_features = notes_with_features.merge(style, on="noteId", how="left")
        if os.path.exists(tox_path):
            tox = pd.read_csv(tox_path)[["noteId", "toxicity"]]
            tox["noteId"] = tox["noteId"].astype(notes_with_features["noteId"].dtype)
            notes_with_features = notes_with_features.merge(tox, on="noteId", how="left")

        _report("#### Readability (Flesch-Kincaid grade level, URLs removed)")
        _report_two_sample(_two_sample("flesch_kincaid_grade", "flesch_kincaid_grade"))
        _report("")
        _report("#### Valence (VADER compound score, URLs removed)")
        _report_two_sample(_two_sample("vader_compound", "vader_compound"), fmt=".3f")
        _report("")
        if "toxicity" in notes_with_features.columns:
            _report("#### Toxicity (Detoxify unbiased checkpoint, URLs removed)")
            _report_two_sample(_two_sample("toxicity", "toxicity"), fmt=".4f")
            _report("")
    else:
        _report(
            "Readability/valence/toxicity: `data/note_text_features.csv` not "
            "found. Run `python note_text_style_features.py` (and "
            "`python detoxify_toxicity.py` for toxicity) to generate it."
        )
        _report("")

    # Top domains
    from urllib.parse import urlparse

    def _extract_domains(text):
        if pd.isna(text):
            return []
        urls = re.findall(r"https?://[^\s]+", str(text))
        domains = []
        for u in urls:
            try:
                parsed = urlparse(u)
                d = parsed.netloc.replace("www.", "")
                if d:
                    if d == "x.com" and "grok" in u.lower():
                        d = "x.com/grok"
                    domains.append(d)
            except Exception:
                continue
        return domains

    notes_with_features["_domains"] = notes_with_features["summary"].apply(
        _extract_domains
    )

    def _normalize_domain(d: str) -> str:
        return "youtube.com" if d in ("youtu.be", "youtube.com") else d

    # Build per-note domain sets
    bot_domain_sets: dict = {}
    human_domain_sets: dict = {}
    for _, row in notes_with_features.iterrows():
        domains_set = set(_normalize_domain(d) for d in row["_domains"])
        if row["writer"] == "bot":
            bot_domain_sets[row["noteId"]] = domains_set
        else:
            human_domain_sets[row["noteId"]] = domains_set

    n_bot_notes = len(bot_domain_sets)
    n_human_notes = len(human_domain_sets)

    bot_domain_counts: Counter = Counter()
    for domains in bot_domain_sets.values():
        for d in domains:
            bot_domain_counts[d] += 1

    human_domain_counts: Counter = Counter()
    for domains in human_domain_sets.values():
        for d in domains:
            human_domain_counts[d] += 1

    top_bot = bot_domain_counts.most_common(10)
    top_human = human_domain_counts.most_common(10)

    def _pct_bot(d):
        return bot_domain_counts.get(d, 0) / n_bot_notes * 100 if n_bot_notes else 0

    def _pct_human(d):
        return (
            human_domain_counts.get(d, 0) / n_human_notes * 100 if n_human_notes else 0
        )

    _report("\n#### Source citation: Top domains by LLM vs human notes")
    _report("")
    _report("**Top 10 domains in LLM notes:**")
    _report("| Rank | Domain | % LLM notes citing | % Human notes citing |")
    _report("|---|---|---|---|")
    for rank, (domain, _) in enumerate(top_bot, 1):
        _report(
            f"| {rank} | {domain} | {_pct_bot(domain):.1f}% | {_pct_human(domain):.1f}% |"
        )

    _report("")
    _report("**Top 10 domains in human notes:**")
    _report("| Rank | Domain | % LLM notes citing | % Human notes citing |")
    _report("|---|---|---|---|")
    for rank, (domain, _) in enumerate(top_human, 1):
        _report(
            f"| {rank} | {domain} | {_pct_bot(domain):.1f}% | {_pct_human(domain):.1f}% |"
        )

    _report("")


# =============================================================================
# CRH rate and hit rate analysis
# =============================================================================


def CRH_rate_analysis(notes_df: pd.DataFrame):
    """
    Compare the bot's CRH rate and hit rate to the distribution across
    individual human writers.
    """
    _report("### Writer percentile benchmarking: CRH rate")
    _report("(Human writers: all from notes-00000.tsv, excluding API authors)")
    _report("")

    human_crh_hit_rate = pd.read_csv(os.path.join(DATA_DIR, "human_crh_hit_rate.csv"))

    # Bot notes
    bot_notes = notes_df[notes_df["writer"] == "bot"]

    # CRH rate
    bot_crh_rate = (
        (bot_notes["finalRatingStatus"] == "CURRENTLY_RATED_HELPFUL").mean() * 100.0
        if len(bot_notes)
        else np.nan
    )
    human_crh_rates = human_crh_hit_rate["n_crh"] / human_crh_hit_rate["n_total"] * 100.0
    percentile = (
        (human_crh_rates <= bot_crh_rate).mean() * 100.0
        if len(human_crh_rates) > 0 and not np.isnan(bot_crh_rate)
        else np.nan
    )
    _report(
        f"Bot CRH rate: {bot_crh_rate:.2f}% "
        f"(percentile among human writers: {percentile:.1f}%)"
    )

    # Hit rate: (#CRH - #CRNH) / total_notes
    _report("\n### Writer percentile benchmarking: hit rate (#CRH - #CRNH) / total notes")
    _report("")
    
    human_hit_rates = (human_crh_hit_rate["n_crh"] - human_crh_hit_rate["n_crnh"]) / human_crh_hit_rate["n_total"] * 100.0
    bot_hit_rate = (
        (
            (bot_notes["finalRatingStatus"] == "CURRENTLY_RATED_HELPFUL").sum()
            - (bot_notes["finalRatingStatus"] == "CURRENTLY_RATED_NOT_HELPFUL").sum()
        )
        / len(bot_notes)
        * 100.0
        if len(bot_notes)
        else np.nan
    )
    hit_rate_percentile = (
        (human_hit_rates <= bot_hit_rate).mean() * 100.0
        if len(human_hit_rates) > 0 and not np.isnan(bot_hit_rate)
        else np.nan
    )
    _report(
        f"Bot hit rate: {bot_hit_rate:.2f}% "
        f"(percentile among human writers: {hit_rate_percentile:.1f}%)"
    )

    # Robustness by human writer note count
    for min_notes in [10, 30]:
        writers_sub = human_crh_hit_rate[human_crh_hit_rate["n_total"] >= min_notes].index
        crh_sub = human_crh_rates[human_crh_rates.index.isin(writers_sub)]
        hit_sub = human_hit_rates[human_hit_rates.index.isin(writers_sub)]
        p_crh = (crh_sub <= bot_crh_rate).mean() * 100 if len(crh_sub) else np.nan
        p_hit = (hit_sub <= bot_hit_rate).mean() * 100 if len(hit_sub) else np.nan
        _report(
            f"Robustness (human writers with >= {min_notes} notes): "
            f"CRH percentile={p_crh:.1f}%, hit rate percentile={p_hit:.1f}% "
            f"(n={len(writers_sub):,} writers)"
        )

    _report("")


# =============================================================================
# common-rater analysis
# =============================================================================


def add_internal_rating_status(note_params: pd.DataFrame) -> pd.DataFrame:
    """
    Add internalRatingStatus column to noteParams based on intercept and factor1.

    Rules:
    - CURRENTLY_RATED_HELPFUL: intercept >= 0.4 AND abs(factor1) < 0.5
    - CURRENTLY_RATED_NOT_HELPFUL: intercept <= -0.05 - 0.8 * abs(factor1)
    - NEED_MORE_RATINGS: everything else
    """
    intercept_col = next(
        (c for c in note_params.columns if c.lower() == "internalnoteintercept"),
        "internalNoteIntercept",
    )
    factor_col = next(
        (
            c
            for c in note_params.columns
            if c.lower() in ("internalnotefactor1", "factor1")
        ),
        None,
    )
    if intercept_col not in note_params.columns:
        raise ValueError(
            f"noteParams must have internalNoteIntercept. Found: {list(note_params.columns)}"
        )
    if factor_col is None:
        raise ValueError(
            f"noteParams must have internalNoteFactor1 or factor1. Found: {list(note_params.columns)}"
        )

    intercept = note_params[intercept_col].to_numpy()
    abs_factor1 = np.abs(note_params[factor_col].to_numpy())

    crh = (intercept >= 0.4) & (abs_factor1 < 0.5)
    crnh = intercept <= -0.05 - 0.8 * abs_factor1

    note_params["internalRatingStatus"] = np.where(
        crh,
        "CURRENTLY_RATED_HELPFUL",
        np.where(crnh, "CURRENTLY_RATED_NOT_HELPFUL", "NEED_MORE_RATINGS"),
    )
    return note_params


def common_raters_note_intercept_analysis(
    notes_df: pd.DataFrame,
    note_params_path: str | None = None,
    min_num_ratings: int = 5,
) -> tuple[dict | None, dict | None]:
    """
    Run t-test and Mann-Whitney U on note intercept (from noteParams)
    for bot vs human notes using common-rater estimates.

    Returns (two_sample_result, lmm_result) for deferred reporting.
    """
    if note_params_path is None:
        note_params_path = os.path.join(DATA_DIR, "noteParams.tsv")
    if not os.path.exists(note_params_path):
        _report(f"  (Skipping: noteParams not found at {note_params_path})")
        return None, None

    note_params = pd.read_csv(note_params_path, sep="\t", low_memory=False)
    add_internal_rating_status(note_params)
    intercept_col = next(
        (c for c in note_params.columns if c.lower() == "internalnoteintercept"), None
    )
    if intercept_col is None:
        _report("  (Skipping: no internalNoteIntercept column)")
        return None, None

    num_ratings_col = next(
        (c for c in note_params.columns if c.lower() == "numratings"), "numRatings"
    )
    note_params = note_params[note_params[num_ratings_col] >= min_num_ratings].copy()
    notes_with_params = notes_df.merge(
        note_params[["noteId", intercept_col, num_ratings_col]].rename(
            columns={intercept_col: "internalNoteIntercept"}
        ),
        on="noteId",
        how="inner",
    )
    res = _two_sample_tests(
        notes_with_params,
        "internalNoteIntercept",
        "internalNoteIntercept (common raters)",
    )
    lmm_res = _run_note_intercept_lmm(
        notes_with_params, "internalNoteIntercept", "Common raters"
    )
    return res, lmm_res


def common_raters_crh_crnh_analysis(
    notes_df: pd.DataFrame,
    note_params_path: str | None = None,
) -> tuple[list[dict], list[dict | None]]:
    """
    Compute %CRH and %CRNH for bot vs human using internalRatingStatus
    from common-rater noteParams.

    Returns (z_test_results, lmm_results) for deferred reporting with BH adjustment.
    Each z_test_results entry has keys: status_name, bot_rate, human_rate, z_stat, z_p.
    """
    if note_params_path is None:
        note_params_path = os.path.join(DATA_DIR, "noteParams.tsv")
    if not os.path.exists(note_params_path):
        _report("  (Skipping %CRH/%CRNH: noteParams not found)")
        return [], []

    note_params = pd.read_csv(note_params_path, sep="\t", low_memory=False)
    add_internal_rating_status(note_params)
    if "internalRatingStatus" not in note_params.columns:
        _report("  (Skipping: internalRatingStatus not in noteParams)")
        return [], []

    notes_with_status = notes_df.merge(
        note_params[["noteId", "internalRatingStatus"]], on="noteId", how="inner"
    )

    _report("\n### %CRH and %CRNH (common-raters noteParams)")
    _report("")
    status_cross = pd.crosstab(
        notes_with_status["writer"],
        notes_with_status["internalRatingStatus"],
        margins=True,
    )
    _report("```")
    _report(status_cross.to_string())
    _report("```")
    _report("")

    z_test_results = []
    for status_name, status_val in [
        ("%CRH", "CURRENTLY_RATED_HELPFUL"),
        ("%CRNH", "CURRENTLY_RATED_NOT_HELPFUL"),
    ]:
        bot_notes = notes_with_status[notes_with_status["writer"] == "bot"]
        human_notes = notes_with_status[notes_with_status["writer"] == "human"]
        bot_rate = (
            (bot_notes["internalRatingStatus"] == status_val).mean() * 100
            if len(bot_notes)
            else np.nan
        )
        human_rate = (
            (human_notes["internalRatingStatus"] == status_val).mean() * 100
            if len(human_notes)
            else np.nan
        )
        bot_count = (bot_notes["internalRatingStatus"] == status_val).sum()
        human_count = (human_notes["internalRatingStatus"] == status_val).sum()
        z_stat, z_p = np.nan, np.nan
        if len(bot_notes) > 0 and len(human_notes) > 0:
            z_stat, z_p = proportions_ztest(
                count=[bot_count, human_count],
                nobs=[len(bot_notes), len(human_notes)],
                alternative="two-sided",
            )
        z_test_results.append({
            "status_name": status_name,
            "bot_rate": bot_rate,
            "human_rate": human_rate,
            "z_stat": z_stat,
            "z_p": z_p,
        })

    # LMMs
    res_crh = _run_status_binary_lmm(
        notes_with_status,
        "internalRatingStatus",
        "CURRENTLY_RATED_HELPFUL",
        "Common raters",
    )
    res_crnh = _run_status_binary_lmm(
        notes_with_status,
        "internalRatingStatus",
        "CURRENTLY_RATED_NOT_HELPFUL",
        "Common raters",
    )

    return z_test_results, [res_crh, res_crnh]


def run_common_raters_analyses(
    fast_start: bool = True,
    note_params_path: str | None = None,
) -> None:
    """
    common-rater analysis: note-level outcomes computed from common raters
    only, using the note parameters (noteParams.tsv) obtained by re-running
    the Community Notes scorer on complete_rater_ratings.csv. Writes the
    'common-rater' report section.
    """
    notes_df, ratings_df = prepare_and_load_data(fast_start=fast_start)

    print("\nFiltering to common raters...")
    ratings_filtered = filter_to_common_raters(ratings_df, notes_df)
    tweets_with_human = notes_df[notes_df["writer"] == "human"]["tweetId"].unique()
    notes_df_subset = notes_df[notes_df["tweetId"].isin(tweets_with_human)]
    _report(
        f"Common raters: {len(ratings_filtered):,} ratings, "
        f"{ratings_filtered['noteId'].nunique():,} notes, "
        f"{ratings_filtered['tweetId'].nunique():,} tweets, "
        f"{ratings_filtered['raterParticipantId'].nunique():,} raters"
    )
    _report("")

    note_params = note_params_path or os.path.join(DATA_DIR, "noteParams.tsv")

    # Collect results from both analyses
    intercept_res, intercept_lmm_res = common_raters_note_intercept_analysis(
        notes_df_subset, note_params_path=note_params
    )
    z_test_results, crh_crnh_lmm_results = common_raters_crh_crnh_analysis(
        notes_df_subset, note_params_path=note_params
    )

    # BH adjustment across the three LMMs (CRH, CRNH, note intercept)
    p_lmm = np.array(
        [
            crh_crnh_lmm_results[0]["p"]
            if len(crh_crnh_lmm_results) > 0 and crh_crnh_lmm_results[0]
            else np.nan,
            crh_crnh_lmm_results[1]["p"]
            if len(crh_crnh_lmm_results) > 1 and crh_crnh_lmm_results[1]
            else np.nan,
            intercept_lmm_res["p"] if intercept_lmm_res else np.nan,
        ]
    )
    p_adj_lmm = _bh_adjust(p_lmm)

    # Report proportion rates (descriptive only)
    for sr in z_test_results:
        _report(
            f"  {sr['status_name']}: LLM={sr['bot_rate']:.2f}%  "
            f"Human={sr['human_rate']:.2f}%"
        )

    # Report intercept descriptive stats (no t-test/U-test)
    if intercept_res:
        _report("\n--- Note intercept (internalNoteIntercept, common raters) ---")
        _report_two_sample(intercept_res, skip_t=True)

    # Report LMMs with BH-adjusted p-values
    _report_lmm_result(
        "CRH",
        crh_crnh_lmm_results[0] if len(crh_crnh_lmm_results) > 0 else None,
        "Common raters",
        p_adj_lmm[0],
    )
    _report_lmm_result(
        "CRNH",
        crh_crnh_lmm_results[1] if len(crh_crnh_lmm_results) > 1 else None,
        "Common raters",
        p_adj_lmm[1],
    )
    _report_lmm_result(
        "Note intercept", intercept_lmm_res, "Common raters", p_adj_lmm[2]
    )

    _flush_section("common-rater")


# =============================================================================
# Representativeness of common raters
# =============================================================================


def rater_distribution_comparison(fast_start: bool = True) -> None:
    """
    Compare coreRaterIntercept and coreRaterFactor1 distributions between the
    full rater population and the common-rater subset. Writes the
    'representativeness' report section.
    """
    print("=" * 80)
    print("RATER DISTRIBUTION: Full Sample vs. Common Raters")
    print("=" * 80)

    notes_df, ratings_df = prepare_and_load_data(fast_start=fast_start)

    full_rater_ids = ratings_df["raterParticipantId"].unique()
    ratings_filtered = filter_to_common_raters(ratings_df, notes_df)
    common_rater_ids = ratings_filtered["raterParticipantId"].unique()

    help_cols = ["raterParticipantId", "coreRaterIntercept", "coreRaterFactor1"]
    helpfulness_sub = ratings_df[help_cols].drop_duplicates()

    full_raters = helpfulness_sub[
        helpfulness_sub["raterParticipantId"].isin(full_rater_ids)
    ].copy()
    common_raters = helpfulness_sub[
        helpfulness_sub["raterParticipantId"].isin(common_rater_ids)
    ].copy()

    _report(
        f"Full sample raters: {len(full_raters):,}; "
        f"common raters: {len(common_raters):,}."
    )
    _report(
        "Distributions of coreRaterIntercept (helpfulness leniency) and "
        "coreRaterFactor1 (political leaning) are compared in "
        "`rater_distribution_full_vs_complete_raters.png`. "
        "See the topic-distribution section for the topic representativeness "
        "chi-square test."
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title in [
        (axes[0], "coreRaterIntercept", "coreRaterIntercept"),
        (axes[1], "coreRaterFactor1", "coreRaterFactor1"),
    ]:
        for df, label, alpha in [
            (full_raters, "Full sample", 0.5),
            (common_raters, "Common raters", 0.5),
        ]:
            vals = df[col].dropna()
            if len(vals) > 0:
                ax.hist(
                    vals,
                    bins=40,
                    alpha=alpha,
                    label=f"{label} (n={len(vals):,})",
                    density=True,
                )
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    out_path = os.path.join(
        OUTPUT_DIR, "rater_distribution_full_vs_complete_raters.png"
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved histogram to {out_path}")
    _flush_section("representativeness")


# =============================================================================
# Within-rater pairwise Bradley-Terry
# =============================================================================

SCORE_MAP = {"HELPFUL": 1.0, "SOMEWHAT_HELPFUL": 0.5, "NOT_HELPFUL": 0.0}


def _fix_pair_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure pair_id is unique per (ai_note_id, human_note_id, tweet_id)."""
    key = ["tweet_id", "ai_note_id", "human_note_id"]
    df = df.copy()
    mapping = df.drop_duplicates(key).reset_index(drop=True)
    mapping = mapping.reset_index().rename(columns={"index": "pair_idx"})
    mapping = mapping[key + ["pair_idx"]]
    df = df.merge(mapping, on=key, how="left")
    df["pair_id"] = "p" + df["pair_idx"].astype(str)
    df = df.drop(columns=["pair_idx"])
    return df


def build_pair_centric_comparisons(
    notes_df: pd.DataFrame,
    ratings_df: pd.DataFrame
) -> pd.DataFrame:
    """
    For each tweet with at least one AI and one human note, enumerate all
    (AI, human) pairs. For each pair, find raters who rated BOTH notes.
    Each (rater, pair) yields one observation: 1 (AI win), 0.5 (tie), 0 (human win).
    """
    ratings_df = ratings_df.merge(
        notes_df[["noteId", "tweetId", "writer"]], on="noteId", how="inner"
    )
    
    ratings_df["rating_score"] = ratings_df["helpfulnessLevel"].map(SCORE_MAP)
    ratings_df = ratings_df.dropna(subset=["rating_score", "coreRaterFactor1"])

    rows = []
    for tweet_id in notes_df["tweetId"].unique():
        tweet_notes = notes_df[notes_df["tweetId"] == tweet_id]
        ai_notes = tweet_notes[tweet_notes["writer"] == "bot"]
        human_notes = tweet_notes[tweet_notes["writer"] == "human"]

        if len(ai_notes) == 0 or len(human_notes) == 0:
            continue

        for _, ai_row in ai_notes.iterrows():
            for _, human_row in human_notes.iterrows():
                ai_note_id = ai_row["noteId"]
                human_note_id = human_row["noteId"]

                ai_ratings = ratings_df[ratings_df["noteId"] == ai_note_id][
                    ["raterParticipantId", "rating_score", "coreRaterFactor1"]
                ].rename(columns={"rating_score": "ai_score"})
                human_ratings = ratings_df[
                    ratings_df["noteId"] == human_note_id
                ][["raterParticipantId", "rating_score"]].rename(
                    columns={"rating_score": "human_score"}
                )

                both = ai_ratings.merge(
                    human_ratings, on="raterParticipantId", how="inner"
                )
                for _, r in both.iterrows():
                    ai_s = r["ai_score"]
                    hu_s = r["human_score"]
                    if ai_s > hu_s:
                        outcome = 1.0
                    elif ai_s < hu_s:
                        outcome = 0.0
                    else:
                        outcome = 0.5
                    rows.append(
                        {
                            "tweet_id": tweet_id,
                            "ai_note_id": ai_note_id,
                            "human_note_id": human_note_id,
                            "rater_id": r["raterParticipantId"],
                            "outcome": outcome,
                            "coreRaterFactor1": r["coreRaterFactor1"],
                        }
                    )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = _fix_pair_ids(df)
    return df






def _run_bradley_terry(df: pd.DataFrame) -> dict | None:
    """
    Bradley-Terry with AI authorship dummy.
    Exclude ties. Fit via logistic regression with SEs clustered by rater.
    """
    bt_df = df[df["outcome"] != 0.5].copy()
    if len(bt_df) < 5:
        _report("Bradley-Terry: insufficient non-tie observations.")
        return None

    y = bt_df["outcome"].values.astype(float)
    X = np.ones((len(y), 1))

    try:
        logit = sm.Logit(y, X)
        result_naive = logit.fit(disp=0)
        result_clustered = logit.fit(
            cov_type="cluster",
            cov_kwds={"groups": bt_df["rater_id"].values},
            disp=0,
        )
        beta = float(result_naive.params[0])
        return {
            "beta_AI": beta,
            "se_naive": float(result_naive.bse[0]),
            "se_clustered": float(result_clustered.bse[0]),
            "n": len(bt_df),
            "n_ties_excluded": len(df) - len(bt_df),
            "result_clustered": result_clustered,
        }
    except Exception as e:
        _report(f"Bradley-Terry fit failed: {e}")
        return None


def run_pairwise_bt_analysis(
    notes_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
) -> pd.DataFrame | None:
    """
    Run pair-centric pairwise comparison with Bradley-Terry model.

    For each tweet with both AI and human notes, enumerates all (AI, human)
    note pairs. For each pair, restricts to raters who rated BOTH notes.
    Fits Bradley-Terry models.
    """
    _report(
        f"Notes: {len(notes_df):,} "
        f"(bot: {(notes_df['writer'] == 'bot').sum():,}, "
        f"human: {(notes_df['writer'] == 'human').sum():,})"
    )
    _report(f"Ratings: {len(ratings_df):,}")
    _report("")

    df = build_pair_centric_comparisons(notes_df, ratings_df)

    if df.empty:
        _report("No pair-centric observations. Exiting.")
        return None

    _report("### Sample sizes")
    _report(f"  Total (rater, pair) observations: {len(df):,}")
    _report(f"  Unique pairs: {df['pair_id'].nunique():,}")
    _report(f"  Unique raters: {df['rater_id'].nunique():,}")
    _report("")

    df["pair_id"] = df["pair_id"].astype(str)
    df["rater_id"] = df["rater_id"].astype(str)

    # Outcome proportions
    n_total = len(df)
    n_tie = int((df["outcome"] == 0.5).sum())
    n_ai_win = int((df["outcome"] == 1.0).sum())
    n_human_win = int((df["outcome"] == 0.0).sum())
    _report("### Outcome proportions (all rater-pair observations)")
    _report(
        f"  Tie: {n_tie:,} ({n_tie / n_total * 100:.1f}%), "
        f"AI win: {n_ai_win:,} ({n_ai_win / n_total * 100:.1f}%), "
        f"Human win: {n_human_win:,} ({n_human_win / n_total * 100:.1f}%)"
    )
    _report("")

    # Bradley-Terry
    _report("### Bradley-Terry (ties excluded, SEs clustered by rater)")
    res = _run_bradley_terry(df)
    if res:
        _report("```")
        _report(str(res["result_clustered"].summary()))
        _report("```")
        _report("")
        _report(f"beta_AI = {res['beta_AI']:.4f} (SE = {res['se_clustered']:.4f})")
        _report(
            f"exp(beta_AI) = {np.exp(res['beta_AI']):.4f} "
            "(odds multiplier for AI vs. human)"
        )
        tie_rate = res["n_ties_excluded"] / len(df) * 100
        pref_ai = 1 / (1 + np.exp(-res["beta_AI"])) * 100
        _report(
            f"n (non-ties): {res['n']:,}, ties excluded: {res['n_ties_excluded']:,} "
            f"(tie rate {tie_rate:.0f}%)"
        )
        _report(f"AI preferred in {pref_ai:.1f}% of non-tied comparisons")
    _report("")

    return df


# =============================================================================
# Main orchestrator
# =============================================================================


def dataset_summary(notes_df: pd.DataFrame, ratings_df: pd.DataFrame):
    """Dataset counts reported in the Abstract and Data and Methods."""
    n_bot = (notes_df["writer"] == "bot").sum()
    n_human = (notes_df["writer"] == "human").sum()
    bot_tweets = notes_df[notes_df["writer"] == "bot"]["tweetId"]
    multi_llm_posts = (bot_tweets.value_counts() > 1).sum()
    _report(f"Notes: {len(notes_df):,} (LLM: {n_bot:,}, human: {n_human:,})")
    _report(f"Unique tweets: {notes_df['tweetId'].nunique():,}")
    _report(f"Ratings: {len(ratings_df):,}")
    _report(f"Unique raters: {ratings_df['raterParticipantId'].nunique():,}")
    _report(f"Posts with more than one LLM note: {multi_llm_posts:,}")


def run_default_analyses(fast_start: bool = True, analyses: set | None = None):
    """
    Run the full-sample analyses and update the corresponding sections of
    outputs/analysis_report.md.

    Parameters
    ----------
    fast_start : bool
        Load pre-computed CSVs (True) or process from raw TSVs (False).
    analyses : set or None
        Analysis names to run (each maps to one report section). If
        empty/None, run all. Valid: dataset, rating, note, timing, text,
        timing_matched, win_rate, pairwise_bt.
    """
    analyses = analyses or set()
    run_all = len(analyses) == 0

    notes_df, ratings_df = prepare_and_load_data(fast_start=fast_start)

    if run_all or "dataset" in analyses:
        dataset_summary(notes_df, ratings_df)
        _flush_section("dataset")
    if run_all or "rating" in analyses:
        rating_analysis_by_bucket()
        _flush_section("rating-buckets")
    if run_all or "note" in analyses:
        note_level_analysis(notes_df)
        CRH_rate_analysis(notes_df)
        _flush_section("note-level")
    if run_all or "timing_matched" in analyses:
        thresholds_analysis(notes_df)
        timing_matched_analysis(notes_df)
        _flush_section("robustness")
    if run_all or "win_rate" in analyses:
        win_rate_analysis(notes_df)
        _flush_section("win-rate")
    if run_all or "timing" in analyses:
        human_bot_timing_analysis(notes_df)
        _flush_section("timing")
    if run_all or "text" in analyses:
        text_features_analysis(notes_df)
        _flush_section("text-features")
    if run_all or "pairwise_bt" in analyses:
        run_pairwise_bt_analysis(notes_df, ratings_df)
        _flush_section("pairwise-bt")


def run_both_notes_analyses(fast_start: bool = True):
    """
    Restrict to posts with at least one LLM and one human note, then re-run
    the rating-bucket means and the note-level outcomes on that subset.
    Updates the 'both-notes' report section. (The subset AI coefficient and
    subset column come from rating_analysis.R.)
    """
    notes_df, ratings_df = prepare_and_load_data(fast_start=fast_start)

    bot_tweets = set(notes_df[notes_df["writer"] == "bot"]["tweetId"])
    human_tweets = set(notes_df[notes_df["writer"] == "human"]["tweetId"])
    both_tweets = bot_tweets & human_tweets
    n_notes_before, n_ratings_before = len(notes_df), len(ratings_df)
    notes_df = notes_df[notes_df["tweetId"].isin(both_tweets)]
    ratings_df = ratings_df[ratings_df["noteId"].isin(notes_df["noteId"])]
    _report(
        f"Restricted to {len(both_tweets):,} tweets with both note types: "
        f"{len(notes_df):,}/{n_notes_before:,} notes "
        f"(LLM: {(notes_df['writer'] == 'bot').sum():,}, "
        f"human: {(notes_df['writer'] == 'human').sum():,}), "
        f"{len(ratings_df):,}/{n_ratings_before:,} ratings"
    )
    _report("")

    rating_analysis_by_bucket(both_writers_only=True, save_figure=False)
    note_level_analysis(notes_df)
    _flush_section("both-notes")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM vs human notes analysis")
    parser.add_argument(
        "--no-fast-start",
        action="store_true",
        help="Process from raw TSVs (requires the raw Community Notes snapshot)",
    )
    parser.add_argument(
        "--analyze-with-common-raters",
        action="store_true",
        help="common-rater analysis with common raters only",
    )
    parser.add_argument(
        "--rater-distribution",
        action="store_true",
        help="Rater distribution, full vs common raters",
    )
    parser.add_argument(
        "--both-notes-only",
        action="store_true",
        help="Appendix replication on tweets with >=1 LLM and >=1 human note",
    )
    parser.add_argument(
        "--analysis",
        nargs="*",
        choices=[
            "dataset",
            "rating",
            "note",
            "timing",
            "text",
            "timing_matched",
            "pairwise_bt",
        ],
        help="Default-mode analyses to run (default: all)",
    )
    parser.add_argument(
        "--note-params-path",
        type=str,
        default=None,
        help="Path to noteParams.tsv (common-rater note parameters)",
    )
    args = parser.parse_args()

    if args.rater_distribution:
        rater_distribution_comparison(fast_start=not args.no_fast_start)
    elif args.analyze_with_common_raters:
        run_common_raters_analyses(
            fast_start=not args.no_fast_start,
            note_params_path=args.note_params_path,
        )
    elif args.both_notes_only:
        run_both_notes_analyses(fast_start=not args.no_fast_start)
    else:
        run_default_analyses(
            fast_start=not args.no_fast_start,
            analyses=set(args.analysis) if args.analysis else set(),
        )
