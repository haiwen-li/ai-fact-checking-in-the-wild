# AI Fact-Checkers in the Wild — Replication Code

Code and data for replicating the analyses in *"AI Fact-Checking in the Wild: A Field Evaluation of LLM-Written Community Notes on X."*

The study compares LLM-written Community Notes against human-written notes.

All results are collected in a single report, [`outputs/analysis_report.md`](outputs/analysis_report.md). Each script updates its own section(s) of that report, so individual analyses can be re-run independently.

## Quick start

```bash
cd code
pip install pandas numpy scipy statsmodels matplotlib
python run_all.py          # full pipeline; add --skip-r to skip the R models
```

`run_all.py` reruns every analysis from the pre-computed CSVs in `data/` and rebuilds `outputs/analysis_report.md` plus the paper figures. 

## Directory structure

```
code/
├── run_all.py               # Orchestrator: runs the full pipeline, renders R-model tables
├── process_data.py          # Data pipeline (raw snapshot -> analysis CSVs)
├── report_utils.py          # Shared section-based report writer
├── analysis.py              # analysis replication
├── rating_analysis.R        # all mixed effects models are run in R
├── note_text_style_features.py  # Precompute: word count, readability, valence
├── detoxify_toxicity.py     # Precompute: note toxicity via Detoxify
├── note_source_citation_features.py  # Precompute: per-note cited-domain dummies
├── source_quality_analysis.py  # MBFC + domain_pc1 source-quality analyses
├── rater_tags_analysis.py   # Rater-provided tags analysis
├── topic_distribution.py    # Topic representativeness
├── post_coverage_analysis.py  # AI vs. both vs. human-only post comparison
├── data/                    # Pre-computed inputs (see table below)
└── outputs/                 # analysis_report.md, paper figures, model CSVs
```

## Data files

| File | Description |
|------|-------------|
| `all_notes.csv` | Combined LLM (bot) + human notes: `noteId`, `noteAuthorParticipantId`, `createdAtMillis`, `tweetId`, `summary`, `isMediaNote`, `writer`, `finalRatingStatus`, `numRatings`, `coreNoteIntercept`, ... |
| `filtered_ratings.csv` | Ratings filtered to analysis notes and raters with valid helpfulness factors (`coreRaterFactor1`, `coreRaterIntercept`), including rater tag columns |
| `ratings_analysis_df.csv` | Rating-level analysis dataset (`AI` indicator, `rating_score`) used by the mixed models in both Python and R |
| `complete_rater_ratings.csv` | common-rater subset: ratings from raters who rated every note on a tweet |
| `noteParams.tsv` | Note parameters (`internalNoteIntercept`, `internalNoteFactor1`, `numRatings`) from re-running the Community Notes scorer on `complete_rater_ratings.csv` |
| `human_crh_hit_rate.csv` | Per human author: `n_total`, `n_crh`, `n_crnh` (writer-percentile benchmarking) |
| `tweet_subgroups.csv` | Per-tweet `topic` and `multimodal` labels (LLM-assigned; prompt in the paper appendix) |
| `mbfc_credibility.json` | Media Bias/Fact Check source ratings used by `source_quality_analysis.py` |
| `domain_pc1.csv` | Aggregate domain-quality scores (Lin et al. 2023 PNAS Nexus), the robustness check for MBFC |
| `note_text_features.csv` | Word count, URL count, Flesch-Kincaid grade, VADER valence per note (URLs stripped); from `note_text_style_features.py` |
| `note_toxicity.csv` | Detoxify toxicity scores per note (URLs stripped); from `detoxify_toxicity.py` |
| `note_source_citations.csv`, `note_source_citations_columns.json` | Per-note 0/1 dummies for the top-cited domains (SI cited-domains analysis); from `note_source_citation_features.py` |
| `tids_api_retrieved.txt` | Tweet IDs targeted by the LLM writer (one per line) |
| `api_account_ids.json` | LLM writer account participant IDs |
| `tweet_human_notes_only.csv` | inputs of `post_coverage_analysis.py`
| `tweet_subgroups_w_followers.csv` | inputs of `post_coverage_analysis.py`

**Data availability.** The raw Community Notes snapshot (downloaded 02/03/2026 from [the Community Notes data download page](https://communitynotes.x.com/guide/en/under-the-hood/download-data)) is too large to distribute here, so `cndata02032026/` ships empty and `process_data.py --no-fast-start` is only runnable if you download a snapshot yourself. Replication does not require it: every analysis runs from the pre-computed CSVs in `data/` (the default "fast-start" path), which were produced by `process_data.py` (and the precompute scripts above) from that snapshot.

## Setup

**Python** ≥ 3.11 with `pandas numpy scipy statsmodels matplotlib`.

Extra dependencies, only needed to *regenerate* the precomputed feature CSVs from scratch (the fast-start path above ships their outputs, so these are optional for reproducing the report):
- `vaderSentiment`, `textstat` for `note_text_style_features.py`
- `detoxify` for `detoxify_toxicity.py` (downloads the `unitary/unbiased-toxic-roberta` checkpoint)

**R** (for `rating_analysis.R`):

```r
install.packages(c("tidyverse", "lme4", "lmerTest", "clubSandwich", "broom.mixed"))
```

## Usage

```bash
# Everything
python run_all.py [--skip-r]

# Individual pieces (each updates its section of outputs/analysis_report.md)
python analysis.py                              # full-sample analyses
python analysis.py --analysis rating note timing_matched win_rate text pairwise_bt
python analysis.py --both-notes-only            # Appendix E replication
python analysis.py --analyze-with-common-raters   # common-rater (Sec. 3.3)
python analysis.py --rater-distribution         # Appendix D figure
python topic_distribution.py
python source_quality_analysis.py               # MBFC + domain_pc1 robustness check
python rater_tags_analysis.py
Rscript rating_analysis.R                       # Table 1, Figures 2-3, Table A7,
                                                 #   and the SI rating-level models
python first_note_ai_proportion.py              # standalone; prints to stdout

# Regenerate the precomputed feature CSVs (optional; already shipped in data/)
python note_text_style_features.py              # word count, readability, valence
python detoxify_toxicity.py                     # toxicity (add --device mps or cuda)
python note_source_citation_features.py         # per-note cited-domain dummies

# Data pipeline (only needed to rebuild data/ from a raw snapshot)
python process_data.py --no-fast-start --common-raters --human-crh-hit-rate
```

## Paper ↔ code map

| Paper result | Script | Output |
|---|---|---|
| Dataset counts (Abstract, Data and Methods) | `analysis.py` | report § *Dataset summary* |
| Timing and exposure asymmetry (Sec. 3.1) | `analysis.py` (`human_bot_timing_analysis`) | report § *Timing and exposure* |
| Figure 1 + Table A1 rating means by ideology | `analysis.py` (`rating_analysis_by_bucket`) | report § *Ratings by rater ideology*; `outputs/rating_analysis_llm_vs_human_barchart.png` |
| Table 1 mixed models (Eq. 1); Table A5 AI coefficients | `rating_analysis.R` (m1–m4, m1_both) | report § *Rating-level mixed models*; `outputs/m*_coefs.csv` |
| Figures 2–3 + Table A7 subgroup fits (crossed REs) | `rating_analysis.R` | report § *Heterogeneity*; `outputs/hte_modality.png`, `outputs/hte_topic.png`, `outputs/hte_*_R.csv` |
| common-rater note-level analysis (Sec. 3.3) | `analysis.py --analyze-with-common-raters` | report § *common-rater* |
| Note length, URL count, top-10 domains (Sec. 3.4) | `analysis.py` (`text_features_analysis`) | report § *Note length, URLs, and cited domains* |
| Readability, valence, toxicity (Sec. 3.4) | `note_text_style_features.py`, `detoxify_toxicity.py`, read by `analysis.py` (`text_features_analysis`) | report § *Note length, URLs, and cited domains* |
| MBFC source quality (Sec. 3.4) | `source_quality_analysis.py` (`mbfc_analysis`) | report § *MBFC source quality*; `outputs/mbfc_distributions.png` |
| Domain-quality robustness check, `domain_pc1` (Sec. 3.4) | `source_quality_analysis.py` (`domain_quality_analysis`) | report § *Domain-quality robustness check*; `outputs/domain_pc1_distributions.png` |
| Rater-provided tags (Sec. 3.5) | `rater_tags_analysis.py` | report § *Rater-provided tags* |
| Within-rater Bradley–Terry (SI Appendix, within-rater comparison) | `analysis.py` (`run_pairwise_bt_analysis`) | report § *Within-rater pairwise comparison* |
| Temporal dynamics of ratings (SI Appendix, Table `temporal_rating`) | `rating_analysis.R` (m1_time_x) | report § *Temporal dynamics of ratings*; `outputs/m1_time_x_coefs.csv` |
| Textual features as predictors of ratings (SI Appendix, Table `rating_text_features`) | `rating_analysis.R` (m1_style, m1_style_x) | report § *Textual features as predictors of ratings*; `outputs/m1_style*_coefs.csv` |
| Cited-domain associations with ratings (SI Appendix, Table `rating_cited_domains`) | `note_source_citation_features.py` + `rating_analysis.R` (m1_sources, m1_sources_x) | report § *Associations between frequently cited domains and ratings*; `outputs/m1_sources*_coefs.csv` |
| Full-sample note-level outcomes + writer percentiles (SI Appendix, Table A2) | `analysis.py` (`note_level_analysis`, `CRH_rate_analysis`) | report § *Full-sample note-level outcomes* |
| Sensitivity to minimum rating thresholds (SI Appendix, Table A3) | `analysis.py` (`thresholds_analysis`) | report § *Note-level robustness checks* |
| Creation-time-matched robustness (SI Appendix, Table `timing_matched_notelevel`) | `analysis.py` (`timing_matched_analysis`) | report § *Note-level robustness checks* |
| Within-post pairwise comparison of note scores (SI Appendix, win rate) | `analysis.py` (`win_rate_analysis`) | report § *Within-post pairwise comparison of note helpfulness scores* |
| Topic representativeness χ² (SI Appendix, representativeness) | `topic_distribution.py` | report § *Topic distribution* |
| Rater representativeness, Figure A1 (SI Appendix) | `analysis.py --rater-distribution` | `outputs/rater_distribution_full_vs_complete_raters.png` |
| Both-note-posts replication (SI Appendix, Tables `both_notes_*`) | `analysis.py --both-notes-only` | report § *Posts with both note types* |

