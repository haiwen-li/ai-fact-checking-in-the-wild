"""
Realized post coverage across LLM and human writers: how do topic, modality,
and OP follower count differ across tweets carrying human-only, AI-only, and
both note types?

Usage: python post_coverage_analysis.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from process_data import load_analysis_notes
from report_utils import DATA_DIR, OUTPUT_DIR

_REPORT: list[str] = []


def _r(msg: str = ""):
    print(msg)
    _REPORT.append(msg)


TWEET_GROUP_ORDER = ["Human only", "AI only", "AI + Human"]
# Single-hue green ramp (matplotlib's Greens colormap), darkest = most human
# involvement in the notes on that tweet.
GROUP_COLORS = {
    "Human only": "#077331",
    "AI + Human": "#43ac5e",
    "AI only": "#91d28e",
}

FOLLOWER_TIER_ORDER = ["<100K", "100K-1M", ">1M"]
FOLLOWER_TIER_BINS = [0, 100_000, 1_000_000, np.inf]


def load_tweet_group_table() -> tuple[pd.DataFrame, int]:
    """
    Tweet-level topic/modality/OP-follower-count table, labeled by which note
    type(s) the tweet carries: Human only, AI only, or AI + Human.

    tweet_human_notes_only.csv is a disjoint set of tweets with only human
    notes.
    tweet_subgroups_w_followers.csv covers every tweet in the analysis sample.
    """
    human_only = pd.read_csv(os.path.join(DATA_DIR, "tweet_human_notes_only.csv"))
    human_only["group"] = "Human only"

    ai_tweets = pd.read_csv(os.path.join(DATA_DIR, "tweet_subgroups_w_followers.csv"))
    writers = load_analysis_notes().groupby("tweetId")["writer"].apply(set)
    tweet_writers = ai_tweets["tweetId"].map(writers)
    has_bot = tweet_writers.apply(lambda w: isinstance(w, set) and "bot" in w)
    has_human = tweet_writers.apply(lambda w: isinstance(w, set) and "human" in w)

    n_dropped = int((~has_bot).sum())
    ai_tweets = ai_tweets[has_bot].copy()
    ai_tweets["group"] = np.where(has_human[has_bot], "AI + Human", "AI only")

    cols = ["tweetId", "topic", "multimodal", "op_followers_count", "group"]
    combined = pd.concat([human_only[cols], ai_tweets[cols]], ignore_index=True)
    combined["group"] = pd.Categorical(
        combined["group"], categories=TWEET_GROUP_ORDER, ordered=True
    )
    combined["follower_tier"] = pd.cut(
        combined["op_followers_count"],
        bins=FOLLOWER_TIER_BINS,
        labels=FOLLOWER_TIER_ORDER,
    )
    return combined, n_dropped


def tweet_characteristics_analysis():
    """
    How do topic, modality, and OP follower count differ across tweets
    carrying human-only, AI-only, and both note types? Purely descriptive
    (no significance tests): percentage-by-group tables plus a 3-panel
    grouped bar chart.
    """
    combined, n_dropped = load_tweet_group_table()

    sizes = combined["group"].value_counts().reindex(TWEET_GROUP_ORDER)
    _r(
        "Group sizes: "
        + ", ".join(f"{g}: {int(n):,}" for g, n in sizes.items())
    )
    if n_dropped:
        _r(
            f"\n({n_dropped} tweet excluded from AI only / AI + Human: its "
            "only AI note falls before the analysis-sample study window, so "
            "it has no in-sample AI note.)"
        )
    _r("")
    _r(
        "Topic and modality are the LLM-assigned per-tweet labels used "
        "elsewhere in the paper; OP follower count is bucketed into three "
        "tiers. Descriptive only -- no significance tests."
    )

    panels = [
        ("topic", "Topic"),
        ("multimodal", "Modality"),
        ("follower_tier", "OP follower tier"),
    ]
    for col, title in panels:
        sub = combined.dropna(subset=[col])
        ct = pd.crosstab(sub[col], sub["group"])
        pct = pd.crosstab(sub[col], sub["group"], normalize="columns") * 100
        if col == "follower_tier":
            order = [t for t in FOLLOWER_TIER_ORDER if t in ct.index]
        else:
            order = ct.sum(axis=1).sort_values(ascending=False).index

        _r(f"\n### {title}")
        _r("")
        _r("| " + title + " | " + " | ".join(TWEET_GROUP_ORDER) + " |")
        _r("|---" * (len(TWEET_GROUP_ORDER) + 1) + "|")
        for cat in order:
            _r(
                f"| {cat} | "
                + " | ".join(
                    f"{pct.loc[cat, g]:.1f}% ({int(ct.loc[cat, g])})"
                    for g in TWEET_GROUP_ORDER
                )
                + " |"
            )
        if col == "follower_tier":
            n_missing = combined["op_followers_count"].isna().groupby(
                combined["group"], observed=True
            ).sum()
            _r(
                "\nMissing follower count (excluded above): "
                + ", ".join(
                    f"{g}: {int(n_missing.get(g, 0)):,}" for g in TWEET_GROUP_ORDER
                )
            )

    # Figure: 3-panel grouped bar chart, one color per group (fixed order).
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (col, title) in zip(axes, panels):
        sub = combined.dropna(subset=[col])
        pct = pd.crosstab(sub[col], sub["group"], normalize="columns") * 100
        if col == "follower_tier":
            cats = [t for t in FOLLOWER_TIER_ORDER if t in pct.index]
        else:
            cats = pct.sum(axis=1).sort_values(ascending=False).index.tolist()
        pct = pct.loc[cats]

        x = np.arange(len(cats))
        width = 0.25
        for i, g in enumerate(TWEET_GROUP_ORDER):
            ax.bar(x + (i - 1) * width, pct[g], width, label=g, color=GROUP_COLORS[g])
        ax.set_xticks(x)
        rotate = col == "topic"
        ax.set_xticklabels(
            cats,
            rotation=40 if rotate else 0,
            ha="right" if rotate else "center",
            fontsize=8,
        )
        ax.set_ylabel("% of group's tweets", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06),
        fontsize=9, frameon=False,
    )
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "tweet_characteristics_distributions.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    _r(f"\nFigure saved: {os.path.basename(fig_path)}")


if __name__ == "__main__":
    tweet_characteristics_analysis()
