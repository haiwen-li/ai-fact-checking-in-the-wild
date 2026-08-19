"""
Topic distribution comparison across the three post samples used in the paper
(Appendix D and Appendix E representativeness checks):

1. All posts included in the analysis (posts with >=1 note in the filtered
   analysis sample).
2. Posts with at least one AI note and at least one human note (Appendix E).
3. Posts included in the common-rater (equal-exposure) analysis
   (Appendix D).

Topics come from data/tweet_subgroups.csv (LLM-assigned, see paper appendix).
For each subset, a chi-square goodness-of-fit test against the all-posts
topic distribution and the Jensen-Shannon distance are reported. Results are
written to the 'topic-dist' section of outputs/analysis_report.md and to
outputs/topic_distribution.csv.

Usage: python topic_distribution.py
"""

import os

import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

from process_data import load_analysis_notes
from report_utils import DATA_DIR, OUTPUT_DIR, update_report_section


def load_sets() -> dict[str, set]:
    notes = load_analysis_notes()

    all_tweets = set(notes["tweetId"].astype(str))
    bot_tweets = set(notes[notes["writer"] == "bot"]["tweetId"].astype(str))
    human_tweets = set(notes[notes["writer"] == "human"]["tweetId"].astype(str))
    both_tweets = bot_tweets & human_tweets

    cr = pd.read_csv(
        os.path.join(DATA_DIR, "complete_rater_ratings.csv"), usecols=["tweetId"]
    )
    cr_tweets = set(cr["tweetId"].astype(str)) & all_tweets

    return {
        "All posts": all_tweets,
        "Posts w/ both note types": both_tweets,
        "Common-rater posts": cr_tweets,
    }


def main():
    topics = pd.read_csv(os.path.join(DATA_DIR, "tweet_subgroups.csv"))
    topics["tweetId"] = topics["tweetId"].astype(str)
    topic_map = topics.set_index("tweetId")["topic"]

    sets = load_sets()

    rows = []
    for set_name, tweet_ids in sets.items():
        t = topic_map.reindex(list(tweet_ids))
        n_missing = t.isna().sum()
        counts = t.value_counts()
        for topic, c in counts.items():
            rows.append(
                {
                    "set": set_name,
                    "topic": topic,
                    "count": c,
                    "pct": c / counts.sum() * 100,
                    "n_posts": len(tweet_ids),
                    "n_missing_topic": n_missing,
                }
            )
    dist = pd.DataFrame(rows)
    dist_path = os.path.join(OUTPUT_DIR, "topic_distribution.csv")
    dist.to_csv(dist_path, index=False)

    # Pivot for the table
    pct = dist.pivot(index="topic", columns="set", values="pct").fillna(0)
    cnt = dist.pivot(index="topic", columns="set", values="count").fillna(0)
    order = pct["All posts"].sort_values(ascending=False).index
    set_order = ["All posts", "Posts w/ both note types", "Common-rater posts"]
    pct = pct.loc[order, set_order]
    cnt = cnt.loc[order, set_order]

    lines = [
        "N posts: " + ", ".join(f"{s}: {len(ids):,}" for s, ids in sets.items()),
        "",
        "### Percent of posts per topic",
        "",
        "| Topic | " + " | ".join(set_order) + " |",
        "|---" * 4 + "|",
    ]
    for topic in pct.index:
        lines.append(
            f"| {topic} | "
            + " | ".join(
                f"{pct.loc[topic, s]:.1f}% ({int(cnt.loc[topic, s])})"
                for s in set_order
            )
            + " |"
        )

    # Chi-square goodness-of-fit: each subset vs. the all-posts distribution.
    # The sets are nested (subsets of "All posts"), so these tests are
    # descriptive rather than tests of independent samples.
    lines += ["", "### Chi-square goodness-of-fit vs. all-posts distribution", ""]
    base_p = pct["All posts"].to_numpy() / 100
    for s in set_order[1:]:
        obs = cnt[s].to_numpy()
        expected = base_p * obs.sum()
        mask = expected > 0
        chi2, p = stats.chisquare(obs[mask], f_exp=expected[mask])
        jsd = jensenshannon(base_p[mask], obs[mask] / obs.sum())
        lines.append(
            f"- {s}: chi2={chi2:.2f}, df={mask.sum() - 1}, p={p:.4f}, "
            f"JS distance={jsd:.4f}"
        )
    lines.append(
        "\n*Sets are nested subsets, so treat tests as descriptive summaries "
        "of distributional shift rather than independent-sample tests.*"
    )

    path = update_report_section("topic-dist", lines)
    print("\n".join(lines))
    print(f"\nWrote {dist_path}")
    print(f"[section 'topic-dist' updated in {path}]")


if __name__ == "__main__":
    main()
