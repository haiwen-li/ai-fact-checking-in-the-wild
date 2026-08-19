"""
Run the full replication pipeline and build outputs/analysis_report.md.

Usage: python run_all.py [--skip-r]
"""

import argparse
import os
import shutil
import subprocess
import sys

import pandas as pd

from report_utils import BASE_DIR, OUTPUT_DIR, update_report_section

# (term in the R tidy output, display label)
TABLE1_TERMS = [
    ("(Intercept)", "Intercept"),
    ("AI1", "AI"),
    ("coreRaterFactor1", "coreRaterFactor1"),
    ("I(coreRaterFactor1^2)", "coreRaterFactor1^2"),
    ("AI1:coreRaterFactor1", "AI x coreRaterFactor1"),
    ("AI1:I(coreRaterFactor1^2)", "AI x coreRaterFactor1^2"),
    ("rater_groupleft", "Left-leaning rater"),
    ("rater_groupright", "Right-leaning rater"),
    ("AI1:rater_groupleft", "AI x left-leaning rater"),
    ("AI1:rater_groupright", "AI x right-leaning rater"),
]
RE_SD_GROUPS = [
    ("raterParticipantId", "SD (rater)"),
    ("noteId", "SD (note)"),
    ("tweetId", "SD (tweet)"),
    ("Residual", "SD (residual)"),
]


def run_step(cmd: list[str], label: str):
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    subprocess.run(cmd, cwd=BASE_DIR, check=True)


def _stars(p: float) -> str:
    if pd.isna(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def render_rating_models_section():
    """Render the rating-level mixed-model coefficients from the R CSVs."""

    def read(name):
        path = os.path.join(OUTPUT_DIR, f"{name}_coefs.csv")
        return pd.read_csv(path) if os.path.exists(path) else None

    m1, m1_both, m2, m3, m4 = (read(n) for n in ["m1", "m1_both", "m2", "m3", "m4"])
    if m1 is None:
        print("R outputs not found; skipping the rating-models section.")
        return

    def lmm_cell(tab, term):
        row = tab[(tab.get("effect") == "fixed") & (tab["term"] == term)]
        if row.empty:
            return ""
        r = row.iloc[0]
        return f"{r['estimate']:.3f}{_stars(r['p.value'])} ({r['std.error']:.3f})"

    def ols_cell(tab, term):
        row = tab[tab["term"] == term]
        if row.empty:
            return ""
        r = row.iloc[0]
        return f"{r['estimate']:.3f}{_stars(r['p_twoway_cr2'])} ({r['se_twoway_cr2']:.3f})"

    def sd_cell(tab, group):
        row = tab[(tab.get("effect") == "ran_pars") & (tab["group"] == group)]
        return "" if row.empty else f"{row.iloc[0]['estimate']:.3f}"

    models = [
        ("Model 1 (note + rater RE)", m1, lmm_cell),
        ("Model 2 (tweet + rater RE)", m2, lmm_cell),
        ("Model 3 (OLS)", m3, ols_cell),
        ("Model 4 (rater group)", m4, lmm_cell),
    ]

    lines = [
        "Equation 1 estimates from rating_analysis.R. Cells show "
        "coefficient (SE); *p<0.05, **p<0.01, ***p<0.001. Model 3 shows "
        "two-way CR2 standard errors clustered by note and rater; classical "
        "and note-only-clustered SEs are also in outputs/m3_coefs.csv.",
        "",
        "| Term | " + " | ".join(name for name, _, _ in models) + " |",
        "|---" * (len(models) + 1) + "|",
    ]
    for term, label in TABLE1_TERMS:
        cells = [cell_fn(tab, term) for _, tab, cell_fn in models]
        if any(cells):
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    for group, label in RE_SD_GROUPS:
        cells = [
            sd_cell(tab, group) if cell_fn is lmm_cell else ""
            for _, tab, cell_fn in models
        ]
        if any(cells):
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append(
        "| N ratings | "
        + " | ".join(f"{int(tab['n_obs'].iloc[0]):,}" for _, tab, _ in models)
        + " |"
    )

    if m1_both is not None:
        ai_full = m1[(m1["effect"] == "fixed") & (m1["term"] == "AI1")].iloc[0]
        ai_both = m1_both[
            (m1_both["effect"] == "fixed") & (m1_both["term"] == "AI1")
        ].iloc[0]
        lines += [
            "",
            "### AI coefficient, full sample vs. both-note posts",
            "",
            f"- Full sample (m1): {ai_full['estimate']:.3f} "
            f"({ai_full['std.error']:.3f}), n={int(ai_full['n_obs']):,}",
            f"- Both-note posts (m1_both): {ai_both['estimate']:.3f} "
            f"({ai_both['std.error']:.3f}), n={int(ai_both['n_obs']):,}",
        ]

    path = update_report_section("rating-models", lines)
    print(f"[section 'rating-models' updated in {path}]")


def render_hte_section():
    """Render the full-sample vs. both-note-posts subgroup coefficients from the R subgroup CSVs."""

    def read(name):
        path = os.path.join(OUTPUT_DIR, name)
        return pd.read_csv(path).set_index("group") if os.path.exists(path) else None

    tables = {
        ("modality", "full"): read("hte_modality_R.csv"),
        ("modality", "both"): read("hte_modality_both_R.csv"),
        ("topic", "full"): read("hte_topic_R.csv"),
        ("topic", "both"): read("hte_topic_both_R.csv"),
    }
    if tables[("modality", "full")] is None or tables[("topic", "full")] is None:
        print("R subgroup outputs not found; skipping the hte section.")
        return

    def cell(table, g):
        if table is None or g not in table.index:
            return "-"
        r = table.loc[g]
        return f"{r['estimate']:.3f} ({r['std.error']:.3f})"

    lines = [
        "AI main-effect coefficient (SE) from the Equation 1 specification "
        "(crossed note + rater random intercepts) fit within each subgroup "
        "by rating_analysis.R.",
        "",
        "| Subgroup | Full sample | Both-note posts |",
        "|---|---|---|",
    ]
    for group_type in ["modality", "topic"]:
        full = tables[(group_type, "full")]
        both = tables[(group_type, "both")]
        for g in full["estimate"].sort_values(ascending=False).index:
            lines.append(f"| {g} | {cell(full, g)} | {cell(both, g)} |")
    path = update_report_section("hte", lines)
    print(f"[section 'hte' updated in {path}]")


def _lmm_cell(tab: pd.DataFrame, term: str) -> str:
    row = tab[(tab.get("effect") == "fixed") & (tab["term"] == term)]
    if row.empty:
        return ""
    r = row.iloc[0]
    return f"{r['estimate']:.3f}{_stars(r['p.value'])} ({r['std.error']:.3f})"


def _read_coefs(name: str):
    path = os.path.join(OUTPUT_DIR, f"{name}_coefs.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


TEMPORAL_TERMS = [
    ("AI1", "AI"),
    ("coreRaterFactor1", "Rater ideology"),
    ("I(coreRaterFactor1^2)", "Rater ideology^2"),
    ("log_time_since_note_c", "Centered log time since note creation"),
    ("AI1:coreRaterFactor1", "AI x rater ideology"),
    ("AI1:I(coreRaterFactor1^2)", "AI x rater ideology^2"),
    ("AI1:log_time_since_note_c", "AI x centered log time"),
]


def render_temporal_section():
    """Render the temporal-dynamics table from m1_time_x_coefs.csv."""
    tab = _read_coefs("m1_time_x")
    if tab is None:
        print("m1_time_x_coefs.csv not found; skipping the rating-temporal section.")
        return

    lines = [
        "Model 1 (Eq. 1 of the main text) extended with the mean-centered "
        "log-transformed elapsed time between note creation and rating "
        "submission and its interaction with AI. Fit by rating_analysis.R "
        "(m1_time_x).",
        "",
        "| Term | Coefficient | SE | p |",
        "|---|---:|---:|---:|",
    ]
    for term, label in TEMPORAL_TERMS:
        row = tab[(tab["effect"] == "fixed") & (tab["term"] == term)]
        if row.empty:
            continue
        r = row.iloc[0]
        p = "<0.001" if r["p.value"] < 0.001 else f"{r['p.value']:.3f}"
        lines.append(f"| {label} | {r['estimate']:.3f} | {r['std.error']:.3f} | {p} |")
    n_obs = int(tab["n_obs"].iloc[0])
    lines.append(f"| N ratings | {n_obs:,} | | |")

    path = update_report_section("rating-temporal", lines)
    print(f"[section 'rating-temporal' updated in {path}]")


TEXT_MEDIATION_TERMS = [
    ("(Intercept)", "Intercept"),
    ("AI1", "LLM-written note"),
    ("coreRaterFactor1", "Rater ideology"),
    ("I(coreRaterFactor1^2)", "Rater ideology^2"),
    ("n_words_z", "Standardized word count"),
    ("n_urls_z", "Standardized URL count"),
    ("flesch_kincaid_grade_z", "Standardized Flesch-Kincaid grade"),
    ("toxicity_z", "Standardized toxicity"),
    ("vader_compound_z", "Standardized valence"),
    ("AI1:coreRaterFactor1", "LLM x rater ideology"),
    ("AI1:I(coreRaterFactor1^2)", "LLM x rater ideology^2"),
    ("AI1:n_words_z", "LLM x standardized word count"),
    ("AI1:n_urls_z", "LLM x standardized URL count"),
    ("AI1:flesch_kincaid_grade_z", "LLM x standardized Flesch-Kincaid grade"),
    ("AI1:toxicity_z", "LLM x standardized toxicity"),
    ("AI1:vader_compound_z", "LLM x standardized valence"),
]


def render_text_mediation_section():
    """
    Render the textual-features-as-predictors table from
    m1_style_coefs.csv (additive controls) and m1_style_x_coefs.csv
    (interactions with LLM authorship).
    """
    additive, interacted = _read_coefs("m1_style"), _read_coefs("m1_style_x")
    if additive is None or interacted is None:
        print("m1_style(_x)_coefs.csv not found; skipping the rating-text-mediation section.")
        return

    lines = [
        "Model 1 (Eq. 1 of the main text) extended with standardized note "
        "text features (word count, URL count, Flesch-Kincaid grade, "
        "toxicity, valence; URLs removed before computing linguistic "
        "measures). Fit by rating_analysis.R (m1_style, m1_style_x).",
        "",
        "| Term | Additive controls | Interactions with LLM |",
        "|---|---:|---:|",
    ]
    for term, label in TEXT_MEDIATION_TERMS:
        lines.append(f"| {label} | {_lmm_cell(additive, term)} | {_lmm_cell(interacted, term)} |")
    lines.append(
        f"| N ratings | {int(additive['n_obs'].iloc[0]):,} | "
        f"{int(interacted['n_obs'].iloc[0]):,} |"
    )
    lines.append("")
    lines.append(
        "*p<0.05, **p<0.01, ***p<0.001. Standard errors in parentheses. "
        "18 notes are dropped for being too short for Flesch-Kincaid scoring."
    )

    path = update_report_section("rating-text-mediation", lines)
    print(f"[section 'rating-text-mediation' updated in {path}]")


def render_cited_domains_section():
    """
    Render the cited-domains-as-predictors table from
    m1_sources_coefs.csv (additive domain indicators) and
    m1_sources_x_coefs.csv (interactions with LLM authorship).
    """
    additive, interacted = _read_coefs("m1_sources"), _read_coefs("m1_sources_x")
    if additive is None or interacted is None:
        print("m1_sources(_x)_coefs.csv not found; skipping the rating-cited-domains section.")
        return

    def domain_label(term: str) -> str:
        d = term[len("cites_") :]
        return "x.com/grok" if d == "x_com_grok" else d.replace("_", ".")

    domain_terms = [
        t for t in additive["term"] if isinstance(t, str) and t.startswith("cites_")
    ]

    lines = [
        "Model 1 (Eq. 1 of the main text) extended with one indicator per "
        "cited domain (domains cited by at least 15 notes of each writer "
        "type get an interaction with AI), controlling for the standardized "
        "total URL count. Fit by rating_analysis.R (m1_sources, "
        "m1_sources_x).",
        "",
        "| Term | Additive domain indicators | Interactions with LLM |",
        "|---|---:|---:|",
        f"| LLM-written note | {_lmm_cell(additive, 'AI1')} | {_lmm_cell(interacted, 'AI1')} |",
        f"| Standardized URL count | {_lmm_cell(additive, 'n_urls_z')} | {_lmm_cell(interacted, 'n_urls_z')} |",
    ]
    for term in domain_terms:
        lines.append(
            f"| `{domain_label(term)}` | {_lmm_cell(additive, term)} | "
            f"{_lmm_cell(interacted, term)} |"
        )
    lines.append("")
    lines.append("*Interactions with LLM authorship*")
    lines.append("")
    lines.append("| Term | Interactions with LLM |")
    lines.append("|---|---:|")
    for term in domain_terms:
        int_term = f"AI1:{term}"
        cell = _lmm_cell(interacted, int_term)
        if cell:
            lines.append(f"| LLM x `{domain_label(term)}` | {cell} |")
    lines.append("")
    lines.append(
        f"N ratings: {int(additive['n_obs'].iloc[0]):,} (additive) / "
        f"{int(interacted['n_obs'].iloc[0]):,} (interacted)."
    )

    path = update_report_section("rating-cited-domains", lines)
    print(f"[section 'rating-cited-domains' updated in {path}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-r",
        action="store_true",
        help="Skip the R mixed models",
    )
    args = parser.parse_args()

    py = sys.executable
    run_step([py, "analysis.py"], "analysis.py (full-sample sections)")
    run_step([py, "analysis.py", "--both-notes-only"], "analysis.py --both-notes-only")
    run_step(
        [py, "analysis.py", "--analyze-with-common-raters"],
        "analysis.py --analyze-with-common-raters",
    )
    run_step(
        [py, "analysis.py", "--rater-distribution"], "analysis.py --rater-distribution"
    )
    run_step([py, "topic_distribution.py"], "topic_distribution.py")
    run_step(
        [py, "source_quality_analysis.py"],
        "source_quality_analysis.py (MBFC + domain_pc1 robustness check)",
    )
    run_step([py, "rater_tags_analysis.py"], "rater_tags_analysis.py")

    if args.skip_r:
        print("\n--skip-r passed: skipping rating_analysis.R and its report sections.")
    elif shutil.which("Rscript") is None:
        print("\nRscript not found: skipping rating_analysis.R (pass --skip-r to silence).")
    else:
        run_step(["Rscript", "rating_analysis.R"], "rating_analysis.R (HTE, SI models)")
        render_rating_models_section()
        render_hte_section()
        render_temporal_section()
        render_text_mediation_section()
        render_cited_domains_section()

    print("\nDone. See outputs/analysis_report.md")


if __name__ == "__main__":
    main()
