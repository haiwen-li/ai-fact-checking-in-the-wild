"""
Source quality of URLs cited in LLM vs human notes: two independent quality
measures over the same URL-extraction and domain-matching pipeline.

`mbfc` (paper Section 3.4, Figure `mbfc_distributions.png`): Media Bias/Fact
Check (MBFC) ratings from data/mbfc_credibility.json.
- Factual Reporting: Very Low ... Very High, coded 0-5 (higher = more
  factual reporting)
- Credibility: Low / Medium / High, coded 0/1/2
Results are written to the 'mbfc' section of outputs/analysis_report.md.

`domain_quality` (paper Section 3.4, robustness check): continuous 0-1
quality scores from data/domain_pc1.csv, which covers many more domains than
MBFC. This is from paper Lin, H., Lasser, J., Lewandowsky, S., Cole, R.,
Gully, A., Rand, D. G., & Pennycook, G. (2023). High level of correspondence
across different news domain quality rating sets. PNAS nexus, 2(9), pgad286.
Results are written to the 'domain-quality' section of
outputs/analysis_report.md.

Matching: note URL netloc (www stripped, youtu.be -> youtube.com) matched to
source domains; if no exact match, subdomains are stripped one label at a
time (en.wikipedia.org -> wikipedia.org). For MBFC, "DEAD" sources are
dropped and duplicate domains are aggregated (numeric mean, categorical
mode); for domain_pc1, duplicate domains are averaged.

Usage:
    python source_quality_analysis.py                    # both analyses
    python source_quality_analysis.py --analysis mbfc
    python source_quality_analysis.py --analysis domain_quality
"""

import argparse
import json
import os
import re
from collections import Counter
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from process_data import load_analysis_notes
from report_utils import DATA_DIR, OUTPUT_DIR
from report_utils import update_report_section as update_analysis_report_section

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
URL_RE = re.compile(r"https?://[^\s]+")

# The MBFC source data contains occasional typos ("Medum" for "Medium"),
# normalized when loading.
CRED_SCALE = {"Low": 0, "Medium": 1, "High": 2}
CRED_ORDER = ["Low", "Medium", "High"]
FR_ORDER = ["Very Low", "Low", "Mixed", "Mostly Factual", "High", "Very High"]
FR_SCALE = {label: i for i, label in enumerate(FR_ORDER)}

# Major social/UGC platforms that neither quality measure rates, reported as
# shares of unmatched URLs (paper Section 3.4; mirrored for domain_quality).
SELECTED_UNMATCHED_DOMAINS = [
    "x.com",
    "youtube.com",
    "instagram.com",
    "reddit.com",
    "tiktok.com",
]

MBFC_PATH = os.path.join(DATA_DIR, "mbfc_credibility.json")
DOMAIN_PC1_PATH = os.path.join(DATA_DIR, "domain_pc1.csv")

_REPORT: list[str] = []


def _r(msg: str = ""):
    print(msg)
    _REPORT.append(msg)


def _flush(update_fn, section_id: str):
    global _REPORT
    path = update_fn(section_id, _REPORT)
    _REPORT = []
    print(f"\n[section '{section_id}' updated in {path}]")


def _norm_domain(netloc: str) -> str:
    d = netloc.lower().strip()
    d = d[4:] if d.startswith("www.") else d
    return "youtube.com" if d == "youtu.be" else d


def load_mbfc() -> pd.DataFrame:
    """Load MBFC ratings, one aggregated row per source domain."""
    with open(MBFC_PATH, encoding="utf-8-sig") as f:
        raw = json.load(f)
    rows = []
    for x in raw:
        src = str(x.get("Source URL", "")).strip()
        if not src or src.upper() == "DEAD":
            continue
        u = src if src.startswith("http") else "http://" + src
        dom = _norm_domain(urlparse(u).netloc)
        if not dom:
            continue
        cred = str(x.get("Credibility", "")).replace("Medum", "Medium")
        fr = str(x.get("Factual Reporting", ""))
        rows.append(
            {
                "domain": dom,
                "credibility": cred if cred in CRED_ORDER else np.nan,
                "cred_score": CRED_SCALE.get(cred, np.nan),
                "factual_reporting": fr if fr in FR_ORDER else np.nan,
                "factual_reporting_score": FR_SCALE.get(fr, np.nan),
            }
        )
    df = pd.DataFrame(rows)

    def _mode(s):
        s = s.dropna()
        return s.mode().iloc[0] if len(s) else np.nan

    agg = df.groupby("domain").agg(
        cred_score=("cred_score", "mean"),
        credibility=("credibility", _mode),
        factual_reporting=("factual_reporting", _mode),
        factual_reporting_score=("factual_reporting_score", "mean"),
    )
    print(f"MBFC: {len(agg):,} unique domains")
    return agg


def load_domain_pc1() -> pd.DataFrame:
    """Load the domain quality scores (pc1: 0 = low quality, 1 = high quality)."""
    df = pd.read_csv(DOMAIN_PC1_PATH)
    df["domain"] = df["domain"].map(_norm_domain)
    df = df.groupby("domain", as_index=True)["pc1"].mean().to_frame()
    print(f"domain_pc1: {len(df):,} unique domains")
    return df


def extract_urls(notes: pd.DataFrame) -> pd.DataFrame:
    """One row per URL cited in a note (no source match yet)."""
    rows = []
    for _, row in notes.iterrows():
        text = str(row["summary"]) if pd.notna(row["summary"]) else ""
        for u in URL_RE.findall(text):
            dom = _norm_domain(urlparse(u).netloc)
            if not dom:
                continue
            rows.append(
                {
                    "noteId": row["noteId"],
                    "tweetId": row["tweetId"],
                    "writer": row["writer"],
                    "url": u,
                    "domain": dom,
                }
            )
    return pd.DataFrame(rows, columns=["noteId", "tweetId", "writer", "url", "domain"])


def _match_domain_factory(known: set):
    """Match a domain to a known set, stripping subdomains one label at a time."""

    def match(dom: str):
        parts = dom.split(".")
        for i in range(len(parts) - 1):
            cand = ".".join(parts[i:])
            if cand in known:
                return cand
        return None

    return match


def build_url_table(notes: pd.DataFrame, mbfc: pd.DataFrame) -> pd.DataFrame:
    """One row per URL cited in a note, with its MBFC match (if any)."""
    urls = extract_urls(notes)
    match = _match_domain_factory(set(mbfc.index))
    urls["mbfc_domain"] = urls["domain"].map(match)
    return urls.merge(mbfc, left_on="mbfc_domain", right_index=True, how="left")


def build_url_table_pc1(notes: pd.DataFrame, pc1: pd.DataFrame) -> pd.DataFrame:
    """One row per URL cited in a note, with its domain_pc1 match (if any)."""
    urls = extract_urls(notes)
    match = _match_domain_factory(set(pc1.index))
    urls["pc1_domain"] = urls["domain"].map(match)
    return urls.merge(pc1, left_on="pc1_domain", right_index=True, how="left")


def _fmt_pct(num: float, den: float) -> str:
    return "NA" if not den else f"{num / den * 100:.1f}%"


def _writer_label(writer: str) -> str:
    return "LLM" if writer == "bot" else "Human"


def _num_compare(df: pd.DataFrame, col: str, label: str, unit: str):
    a = df.loc[df["writer"] == "bot", col].dropna()
    h = df.loc[df["writer"] == "human", col].dropna()
    t, p = stats.ttest_ind(a, h, equal_var=False)
    u, up = stats.mannwhitneyu(a, h, alternative="two-sided")
    sp = np.sqrt(
        ((len(a) - 1) * a.std() ** 2 + (len(h) - 1) * h.std() ** 2)
        / (len(a) + len(h) - 2)
    )
    d = (a.mean() - h.mean()) / sp if sp > 0 else np.nan
    _r(
        f"| {label} | {a.mean():.3f} | {h.mean():.3f} | {d:.3f} "
        f"| t={t:.2f}, p={p:.2g} | U p={up:.2g} | {len(a):,} / {len(h):,} {unit} |"
    )


def _dist_compare(urls: pd.DataFrame, col: str, order: list[str], label: str):
    sub = urls.dropna(subset=[col])
    ct = pd.crosstab(sub["writer"], sub[col])
    ct = ct[[c for c in order if c in ct.columns]]
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    _r(f"\n#### {label} distribution (% of matched URLs; chi2={chi2:.1f}, p={p:.2g})")
    _r("")
    _r("| Writer | " + " | ".join(ct.columns) + " |")
    _r("|---" * (len(ct.columns) + 1) + "|")
    for w in ["bot", "human"]:
        row = ct.loc[w]
        pcts = row / row.sum() * 100
        _r(f"| {_writer_label(w)} | " + " | ".join(f"{v:.1f}" for v in pcts) + " |")
    return ct


def _domain_count_text(items: list[tuple[str, int]], idx: int) -> str:
    if idx >= len(items):
        return ""
    domain, count = items[idx]
    return f"{domain} ({count:,})"


def coverage_and_unmatched(notes: pd.DataFrame, urls: pd.DataFrame, match_col: str):
    _r("\n### Coverage")
    _r("")
    _r(
        "| Writer | URLs cited | URLs matched | % URLs matched | URL-citing notes | "
        "URL-citing notes with >=1 matched URL | % URL-citing notes matched | "
        "All notes | All notes with >=1 matched URL | % all notes matched |"
    )
    _r("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for w, lab in [("bot", "LLM"), ("human", "Human")]:
        note_sub = notes[notes["writer"] == w]
        url_sub = urls[urls["writer"] == w]
        matched = url_sub[match_col].notna()
        all_notes = note_sub["noteId"].nunique()
        url_notes = url_sub["noteId"].nunique()
        matched_notes = url_sub.loc[matched, "noteId"].nunique()
        _r(
            f"| {lab} | {len(url_sub):,} | {matched.sum():,} | "
            f"{_fmt_pct(matched.sum(), len(url_sub))} | {url_notes:,} | "
            f"{matched_notes:,} | {_fmt_pct(matched_notes, url_notes)} | "
            f"{all_notes:,} | {matched_notes:,} | "
            f"{_fmt_pct(matched_notes, all_notes)} |"
        )

    unmatched = urls[urls[match_col].isna()]
    top_overall = Counter(unmatched["domain"]).most_common(10)
    top_ai = Counter(unmatched.loc[unmatched["writer"] == "bot", "domain"]).most_common(
        10
    )
    top_human = Counter(
        unmatched.loc[unmatched["writer"] == "human", "domain"]
    ).most_common(10)

    _r("\n### Top unmatched domains")
    _r("")
    _r("| Rank | Overall | LLM notes | Human notes |")
    _r("|---:|---|---|---|")
    for i in range(10):
        _r(
            f"| {i + 1} | {_domain_count_text(top_overall, i)} | "
            f"{_domain_count_text(top_ai, i)} | {_domain_count_text(top_human, i)} |"
        )

    _r("\n### Selected unmatched-domain shares")
    _r("")
    _r(
        "| Domain | LLM unmatched URLs | % of LLM unmatched URLs | "
        "Human unmatched URLs | % of human unmatched URLs |"
    )
    _r("|---|---:|---:|---:|---:|")
    writer_unmatched = {
        w: unmatched[unmatched["writer"] == w] for w in ["bot", "human"]
    }
    selected_counts = {"bot": 0, "human": 0}
    for domain in SELECTED_UNMATCHED_DOMAINS:
        counts = {
            w: int((sub["domain"] == domain).sum())
            for w, sub in writer_unmatched.items()
        }
        for w, count in counts.items():
            selected_counts[w] += count
        _r(
            f"| {domain} | {counts['bot']:,} | "
            f"{_fmt_pct(counts['bot'], len(writer_unmatched['bot']))} | "
            f"{counts['human']:,} | "
            f"{_fmt_pct(counts['human'], len(writer_unmatched['human']))} |"
        )
    _r(
        f"| **Selected total** | {selected_counts['bot']:,} | "
        f"{_fmt_pct(selected_counts['bot'], len(writer_unmatched['bot']))} | "
        f"{selected_counts['human']:,} | "
        f"{_fmt_pct(selected_counts['human'], len(writer_unmatched['human']))} |"
    )
    combined = selected_counts["bot"] + selected_counts["human"]
    _r(
        f"\nSelected domains combined: {combined:,} of {len(unmatched):,} "
        f"unmatched URLs ({_fmt_pct(combined, len(unmatched))})"
    )

    # Government and other official institutional domains, which neither
    # quality measure rates (both rate media outlets, not official sources).
    def is_official(d: str) -> bool:
        return (
            d.endswith(".gov")
            or d.endswith(".mil")
            or d.endswith(".int")
            or ".gov." in d
            or d == "gov.uk"
        )

    official = unmatched[unmatched["domain"].map(is_official)]
    n_official_bot = (official["writer"] == "bot").sum()
    n_official_human = (official["writer"] == "human").sum()
    _r(
        f"\nGovernment/official institutional domains (.gov, .mil, .int, "
        f".gov.xx): {len(official):,} of {len(unmatched):,} unmatched URLs "
        f"({_fmt_pct(len(official), len(unmatched))}; LLM {n_official_bot:,}, "
        f"human {n_official_human:,})"
    )


def plot_distributions(cts: list[tuple[pd.DataFrame, list[str], str]], fig_path: str):
    fig, axes = plt.subplots(1, len(cts), figsize=(5.3 * len(cts), 4.6))
    if len(cts) == 1:
        axes = [axes]
    for ax, (ct, order, title) in zip(axes, cts):
        cols = [c for c in order if c in ct.columns]
        x = np.arange(len(cols))
        for i, (w, lab, color) in enumerate(
            [("bot", "LLM", "#a9dfbf"), ("human", "Human", "#27ae60")]
        ):
            if w not in ct.index or not ct.loc[w].sum():
                pcts = pd.Series(0, index=cols)
            else:
                pcts = (ct.loc[w] / ct.loc[w].sum() * 100)[cols]
            ax.bar(x + (i - 0.5) * 0.38, pcts, 0.38, label=lab, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("% of matched URLs")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def mbfc_analysis():
    """Paper Section 3.4: MBFC-rated source quality of cited URLs."""
    mbfc = load_mbfc()
    notes = load_analysis_notes()
    urls = build_url_table(notes, mbfc)

    coverage_and_unmatched(notes, urls, "mbfc_domain")

    matched = urls.dropna(subset=["mbfc_domain"]).copy()

    _r("\n### URL-level summary stats (matched citations)")
    _r("")
    _r("| Measure | Writer | n | mean | sd | median |")
    _r("|---|---|---:|---:|---:|---:|")
    for col, label in [
        ("factual_reporting_score", "Factual reporting (0=Very Low, 5=Very High)"),
        ("cred_score", "Credibility (0=Low,1=Med,2=High)"),
    ]:
        for w in ["bot", "human"]:
            vals = matched.loc[matched["writer"] == w, col].dropna()
            _r(
                f"| {label} | {_writer_label(w)} | {len(vals):,} | "
                f"{vals.mean():.3f} | {vals.std():.3f} | {vals.median():.3f} |"
            )

    _r("\n### URL-level comparison (each matched citation is one observation)")
    _r("")
    _r(
        "| Measure | LLM mean | Human mean | Cohen's d | Welch t | "
        "Mann-Whitney | n (LLM/human) |"
    )
    _r("|---|---|---|---|---|---|---|")
    _num_compare(
        matched,
        "factual_reporting_score",
        "Factual reporting (0=Very Low, 5=Very High)",
        "URLs",
    )
    _num_compare(matched, "cred_score", "Credibility (0=Low,1=Med,2=High)", "URLs")

    ct_cred = _dist_compare(matched, "credibility", CRED_ORDER, "Credibility")
    ct_fr = _dist_compare(matched, "factual_reporting", FR_ORDER, "Factual reporting")

    fig_path = os.path.join(OUTPUT_DIR, "mbfc_distributions.png")
    plot_distributions(
        [(ct_cred, CRED_ORDER, "Credibility"), (ct_fr, FR_ORDER, "Factual reporting")],
        fig_path,
    )
    _r(f"\nFigure saved: {os.path.basename(fig_path)}")

    _flush(update_analysis_report_section, "mbfc")


def domain_quality_analysis():
    """
    R1. The MBFC source-quality analysis, repeated with domain_pc1.csv.

    This is deliberately a strict mirror of the `mbfc` analysis: same URL
    extraction, same domain normalization, same subdomain-stripped matching,
    same tables in the same order. The only substantive change is the quality
    measure, which is continuous here rather than the two ordinal MBFC
    scales, so the ordinal distribution tables have no counterpart and the
    figure is a density rather than a bar chart. Coverage is wider because
    domain_pc1 scores more domains.
    """
    pc1 = load_domain_pc1()
    notes = load_analysis_notes()
    urls = build_url_table_pc1(notes, pc1)

    _r(
        "Source quality scored with `domain_pc1.csv` "
        f"({len(pc1):,} domains; 0 = low quality, 1 = high quality). "
        "This repeats the `mbfc` analysis table for table, changing only "
        "the quality measure. URL extraction, domain normalization and "
        "subdomain-stripped matching are identical, so the coverage figures "
        "are directly comparable with the MBFC section of the main report."
    )
    _r("")
    _r(
        "The MBFC section additionally reports the distribution across the "
        "ordinal Credibility and Factual Reporting categories. `pc1` is a "
        "single continuous score, so those two tables have no counterpart "
        "here."
    )
    _r("")

    coverage_and_unmatched(notes, urls, "pc1_domain")

    matched = urls.dropna(subset=["pc1_domain"]).copy()

    _r("\n### URL-level summary stats (matched citations)")
    _r("")
    _r("| Measure | Writer | n | mean | sd | median |")
    _r("|---|---|---:|---:|---:|---:|")
    for w in ["bot", "human"]:
        vals = matched.loc[matched["writer"] == w, "pc1"].dropna()
        _r(
            f"| Domain quality (0=low, 1=high) | {_writer_label(w)} | "
            f"{len(vals):,} | {vals.mean():.3f} | {vals.std():.3f} | "
            f"{vals.median():.3f} |"
        )

    _r("\n### URL-level comparison (each matched citation is one observation)")
    _r("")
    _r(
        "| Measure | LLM mean | Human mean | Cohen's d | Welch t | "
        "Mann-Whitney | n (LLM/human) |"
    )
    _r("|---|---|---|---|---|---|---|")
    _num_compare(matched, "pc1", "Domain quality (0=low, 1=high)", "URLs")

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for w, lab, color in [("bot", "LLM", "#a9dfbf"), ("human", "Human", "#27ae60")]:
        ax.hist(
            matched.loc[matched["writer"] == w, "pc1"],
            bins=30,
            range=(0, 1),
            alpha=0.6,
            density=True,
            label=lab,
            color=color,
        )
    ax.set_xlabel("Domain quality (0 = low, 1 = high)")
    ax.set_ylabel("Density of matched URLs")
    ax.set_title("Domain quality (pc1)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "domain_pc1_distributions.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    _r(f"\nFigure saved: {os.path.basename(fig_path)}")

    _flush(update_analysis_report_section, "domain-quality")


ANALYSES = {
    "mbfc": mbfc_analysis,
    "domain_quality": domain_quality_analysis,
}


def main():
    parser = argparse.ArgumentParser(
        description="Source-quality analyses (MBFC + domain_pc1)"
    )
    parser.add_argument(
        "--analysis",
        nargs="*",
        choices=sorted(ANALYSES.keys()),
        help="Analyses to run (default: all)",
    )
    args = parser.parse_args()
    names = args.analysis or list(ANALYSES.keys())
    for name in names:
        print("=" * 78)
        print(f"Running: {name}")
        print("=" * 78)
        ANALYSES[name]()


if __name__ == "__main__":
    main()
