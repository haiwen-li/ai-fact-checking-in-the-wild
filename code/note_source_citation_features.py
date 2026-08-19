"""
Per-note citation features for the top-cited-source outcome analysis.

Domain set: the union of the top-10 domains cited by LLM notes and the
top-10 cited by human notes, recomputed here with the same extraction and
normalization analysis.py uses.

Writes data/note_source_citations.csv: one row per note with
    noteId, tweetId, writer, cites_<domain1>, ..., cites_<domainN>
where each cites_* column is 0/1 for whether that note's summary links to
that exact domain.

Usage:
    python note_source_citation_features.py
"""

import json
import os
import re
from collections import Counter
from urllib.parse import urlparse

import pandas as pd

from process_data import load_analysis_notes
from report_utils import DATA_DIR

URL_RE = re.compile(r"https?://[^\s]+")
TOP_N = 10
MIN_CITATIONS_PER_WRITER = 15

OUT_CSV = os.path.join(DATA_DIR, "note_source_citations.csv")
OUT_COLUMNS_JSON = os.path.join(DATA_DIR, "note_source_citations_columns.json")


def _extract_domains(text) -> list[str]:
    """Mirrors analysis.py's text_features_analysis._extract_domains."""
    if pd.isna(text):
        return []
    domains = []
    for u in URL_RE.findall(str(text)):
        try:
            d = urlparse(u).netloc.replace("www.", "")
        except Exception:
            continue
        if not d:
            continue
        if d == "x.com" and "grok" in u.lower():
            d = "x.com/grok"
        domains.append(d)
    return domains


def _normalize_domain(d: str) -> str:
    return "youtube.com" if d in ("youtu.be", "youtube.com") else d


def top_domains_by_writer(notes: pd.DataFrame, n: int = TOP_N) -> dict[str, list[str]]:
    """Top-n domains per writer, one set of domains per note (matches
    analysis.py's Section 3.4 top-10 tables)."""
    counts = {"bot": Counter(), "human": Counter()}
    for _, row in notes.iterrows():
        domains = {_normalize_domain(d) for d in _extract_domains(row["summary"])}
        counts[row["writer"]].update(domains)
    return {w: [d for d, _ in c.most_common(n)] for w, c in counts.items()}


def _sanitize(domain: str) -> str:
    return "cites_" + re.sub(r"[^0-9a-zA-Z]+", "_", domain).strip("_").lower()


def build_citation_matrix(notes: pd.DataFrame, domains: list[str]) -> pd.DataFrame:
    col_of = {d: _sanitize(d) for d in domains}
    rows = []
    for _, row in notes.iterrows():
        note_domains = {_normalize_domain(d) for d in _extract_domains(row["summary"])}
        rec = {
            "noteId": row["noteId"],
            "tweetId": row["tweetId"],
            "writer": row["writer"],
        }
        for d in domains:
            rec[col_of[d]] = int(d in note_domains)
        rows.append(rec)
    df = pd.DataFrame(rows, columns=["noteId", "tweetId", "writer"] + list(col_of.values()))
    return df, col_of


def main():
    notes = load_analysis_notes()
    per_writer = top_domains_by_writer(notes)
    domains = sorted(set(per_writer["bot"]) | set(per_writer["human"]))
    print(f"Top-{TOP_N} LLM domains: {per_writer['bot']}")
    print(f"Top-{TOP_N} human domains: {per_writer['human']}")
    print(f"Union: {len(domains)} unique domains")

    citations, col_of = build_citation_matrix(notes, domains)

    print("\nCitation counts by writer (flag: <15 citations in either group):")
    print(f"{'domain':30s} {'LLM':>8s} {'human':>8s}")
    rare = []
    for d in domains:
        col = col_of[d]
        n_bot = int(citations.loc[citations["writer"] == "bot", col].sum())
        n_human = int(citations.loc[citations["writer"] == "human", col].sum())
        flag = ""
        if n_bot < MIN_CITATIONS_PER_WRITER or n_human < MIN_CITATIONS_PER_WRITER:
            flag = "  <-- rare in at least one writer group"
            rare.append(d)
        print(f"{d:30s} {n_bot:8d} {n_human:8d}{flag}")

    if rare:
        print(
            f"\n{len(rare)} domain(s) rare in at least one writer group: {rare}. "
            "Kept in the output matrix; rating_analysis.R drops these from "
            "m1_sources_x, where the interaction isn't identifiable."
        )

    citations.to_csv(OUT_CSV, index=False)
    with open(OUT_COLUMNS_JSON, "w") as f:
        json.dump(col_of, f, indent=2)
    print(f"\nWrote {OUT_CSV} ({len(citations):,} notes, {len(domains)} domain columns)")
    print(f"Wrote {OUT_COLUMNS_JSON} (column -> domain mapping)")


if __name__ == "__main__":
    main()
