# Rating-level mixed-effects models (paper Table 1, Equation 1) and the
# subgroup heterogeneity fits behind Figures 2-3.
#
# Models (rating_score coded HELPFUL = 1, SOMEWHAT_HELPFUL = 0.5,
# NOT_HELPFUL = 0):
#   m1      Table 1 Model 1: AI * factor + AI * factor^2 +
#           (1|noteId) + (1|raterParticipantId)          [full sample]
#   m1_both m1 restricted to posts with >=1 LLM and >=1 human note
#           (Table A5 subset column)
#   m2      Table 1 Model 2: tweet + rater random intercepts
#   m3      Table 1 Model 3: OLS; both classical and CR2 note-clustered SEs
#           are exported
#   m4      Table 1 Model 4: AI * rater_group (left/middle/right) +
#           note + rater random intercepts
#   HTE     m1 re-fit within each tweet topic / modality subgroup, for the
#           full sample (Figures 2-3) and for the both-note subset
#           (together these are Table A7)
#
# Outputs (written to outputs/):
#   m1_coefs.csv, m1_both_coefs.csv, m2_coefs.csv, m4_coefs.csv
#     broom.mixed::tidy() tables (fixed effects + random-effect SDs) with
#     an n_obs column
#   m3_coefs.csv
#     OLS estimates with classical SEs and clubSandwich CR2 SEs clustered
#     by noteId
#   hte_topic_R.csv, hte_modality_R.csv, hte_topic_both_R.csv,
#   hte_modality_both_R.csv, hte_topic.png, hte_modality.png
#   m1_spec_sensitivity_coefs.csv
#     broom.mixed::tidy() tables for the none/length/cites/length_cites
#     control specifications, stacked with a spec column
#
# Run from anywhere: Rscript rating_analysis.R
# Requires: tidyverse, lme4, lmerTest, clubSandwich, broom.mixed

suppressPackageStartupMessages({
  library(tidyverse)
  library(lmerTest)
  library(lme4)
  library(clubSandwich)
  library(broom.mixed)
})

# Resolve paths relative to this script so no setwd() is needed.
cmd_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", cmd_args, value = TRUE)
base_dir <- if (length(file_arg)) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1])))
} else {
  getwd()  # interactive use: run from the code/ directory
}
data_dir <- file.path(base_dir, "data")
out_dir <- file.path(base_dir, "outputs")
dir.create(out_dir, showWarnings = FALSE)

# Study window start (Nov 1, 2025 ET, epoch ms); must match
# process_data.STUDY_START_MILLIS.
STUDY_START_MILLIS <- 1761969600000

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

ratings_df <- read_csv(
  file.path(data_dir, "ratings_analysis_df.csv"),
  col_types = cols(
    noteId = col_character(),
    ratedOnTweetId = col_character(),
    tweetId = col_character()
  )
)
ratings_df$AI <- as.factor(ratings_df$AI)
ratings_df <- ratings_df %>%
  mutate(
    rater_group = case_when(
      coreRaterFactor1 < -0.15 ~ "left",
      coreRaterFactor1 > 0.15 ~ "right",
      TRUE ~ "middle"
    ),
    rater_group = factor(rater_group, levels = c("middle", "left", "right"))
  )


# Posts with >=1 LLM and >=1 human note, defined by note presence in the
# filtered analysis sample (media notes and pre-study-window notes excluded),
# not by which notes happen to have ratings.
all_notes <- read_csv(
  file.path(data_dir, "all_notes.csv"),
  col_types = cols(noteId = col_character(), tweetId = col_character())
) %>%
  filter(isMediaNote == 0, createdAtMillis > STUDY_START_MILLIS)
tid_both <- intersect(
  unique(all_notes$tweetId[all_notes$writer == "bot"]),
  unique(all_notes$tweetId[all_notes$writer == "human"])
)

ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
form_main <- rating_score ~ AI * coreRaterFactor1 + AI * I(coreRaterFactor1^2) +
  (1 | noteId) + (1 | raterParticipantId)

save_tidy <- function(mod, name) {
  tab <- broom.mixed::tidy(mod) %>% mutate(n_obs = nobs(mod))
  write_csv(tab, file.path(out_dir, paste0(name, "_coefs.csv")))
  message("=== ", name, " (n = ", format(nobs(mod), big.mark = ","), ") ===")
  print(tab, n = Inf)
  tab
}

# ---------------------------------------------------------------------------
# Table 1 models
# ---------------------------------------------------------------------------

message("Fitting m1 (Table 1 Model 1) ...")
m1 <- lmer(form_main, data = ratings_df, control = ctrl)
save_tidy(m1, "m1")

message("Fitting m1_both (Table A5 subset column) ...")
m1_both <- lmer(
  form_main,
  data = ratings_df[ratings_df$tweetId %in% tid_both, ],
  control = ctrl
)
save_tidy(m1_both, "m1_both")

message("Fitting m2 (Table 1 Model 2) ...")
m2 <- lmer(
  rating_score ~ AI * coreRaterFactor1 + AI * I(coreRaterFactor1^2) +
    (1 | tweetId) + (1 | raterParticipantId),
  data = ratings_df,
  control = ctrl
)
save_tidy(m2, "m2")

message("Fitting m3 (Table 1 Model 3, OLS) ...")
m3 <- lm(
  rating_score ~ AI * coreRaterFactor1 + AI * I(coreRaterFactor1^2),
  data = ratings_df
)

inter_id <- paste(ratings_df$noteId, ratings_df$raterParticipantId, sep = "_")

message("CR2 vcov: note ...")
v_note_cr2 <- vcovCR(m3, cluster = ratings_df$noteId, type = "CR2")
message("CR2 vcov: rater ...")
v_rater_cr2 <- vcovCR(m3, cluster = ratings_df$raterParticipantId, type = "CR2")
message("CR2 vcov: note x rater intersection ...")
v_inter_cr2 <- vcovCR(m3, cluster = inter_id, type = "CR2")

m3_note <- as.data.frame(coef_test(m3, vcov = v_note_cr2)) %>%
  rownames_to_column("term") %>%
  transmute(term, se_note_cr2 = SE, t_note_cr2 = tstat, p_note_cr2 = p_Satt)

psd_fix <- function(V, dn) {
  ev <- eigen(V, symmetric = TRUE, only.values = TRUE)$values
  if (any(ev < 0)) {
    warning(
      "Covariance matrix is not positive semi-definite (smallest eigenvalue ",
      signif(min(ev), 3), "). Applying eigenvalue truncation."
    )
    e <- eigen(V, symmetric = TRUE)
    V <- e$vectors %*% diag(pmax(e$values, 0)) %*% t(e$vectors)
    dimnames(V) <- dn
  }
  V
}

# Two-way combination (Cameron, Gelbach & Miller 2011): sum the note- and
# rater-clustered covariance matrices, subtract the intersection. Applied
# here to the CR2 matrices rather than the classical CR0 sandwich.
v_twoway_cr2 <- psd_fix(
  v_note_cr2 + v_rater_cr2 - v_inter_cr2, dimnames(v_note_cr2)
)

se_of <- function(V) sqrt(diag(V))
m3_tab <- broom::tidy(m3) %>%
  rename(se_classical = std.error) %>%
  left_join(m3_note, by = "term") %>%
  mutate(
    se_rater_cr2 = se_of(v_rater_cr2)[term],
    se_twoway_cr2 = se_of(v_twoway_cr2)[term],
    z_twoway_cr2 = estimate / se_twoway_cr2,
    p_twoway_cr2 = 2 * pnorm(-abs(z_twoway_cr2)),
    n_obs = nobs(m3)
  )
write_csv(m3_tab, file.path(out_dir, "m3_coefs.csv"))
message("=== m3_twoway (n = ", format(nobs(m3), big.mark = ","), ") ===")
print(m3_tab, n = Inf)



message("Fitting m4 (Table 1 Model 4) ...")
m4 <- lmer(
  rating_score ~ AI * rater_group + (1 | noteId) + (1 | raterParticipantId),
  data = ratings_df,
  control = ctrl
)
save_tidy(m4, "m4")

# ---------------------------------------------------------------------------
# Subgroup heterogeneity (Figures 2-3)
# ---------------------------------------------------------------------------

tweet_subgroups <- read_csv(
  file.path(data_dir, "tweet_subgroups.csv"),
  col_types = cols(tweetId = col_character())
)
df <- left_join(ratings_df, tweet_subgroups, by = "tweetId")

fit_subgroups <- function(df, group_col) {
  groups <- df %>%
    distinct(.data[[group_col]]) %>%
    filter(!is.na(.data[[group_col]])) %>%
    pull(.data[[group_col]])

  map_dfr(groups, function(g) {
    d <- df %>% filter(.data[[group_col]] == g)
    empty <- tibble(
      group = g, n_obs = nrow(d), n_note = n_distinct(d$noteId),
      n_rater = n_distinct(d$raterParticipantId),
      estimate = NA_real_, std.error = NA_real_
    )
    mod <- tryCatch(lmer(form_main, data = d, control = ctrl),
                    error = function(e) NULL)
    if (is.null(mod)) return(empty)
    # AI main effect (factor-coded, so the term is "AI1")
    ai <- broom.mixed::tidy(mod, effects = "fixed") %>%
      filter(str_detect(term, "^AI"), !str_detect(term, ":")) %>%
      dplyr::slice(1)
    if (nrow(ai) == 0) return(empty)
    empty %>% mutate(estimate = ai$estimate, std.error = ai$std.error)
  }) %>%
    mutate(
      conf.low = estimate - 1.96 * std.error,
      conf.high = estimate + 1.96 * std.error
    ) %>%
    filter(!is.na(estimate)) %>%
    arrange(estimate) %>%
    mutate(group = fct_inorder(group))
}

plot_subgroups <- function(results, y_label, fig_path) {
  p <- ggplot(results, aes(x = estimate, y = group)) +
    theme_bw(base_size = 12) +
    theme(
      panel.grid.major.y = element_line(
        color = "grey80", linetype = "dashed", linewidth = 0.5
      ),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(fill = NA, linewidth = 0.8),
      plot.title = element_text(face = "bold", size = rel(1.05), hjust = 0.5),
      axis.title.y = element_text(margin = margin(r = 8)),
      axis.title.x = element_text(margin = margin(t = 8))
    ) +
    geom_vline(
      xintercept = 0, linetype = "dashed", color = "red", linewidth = 0.7
    ) +
    geom_linerange(
      aes(xmin = conf.low, xmax = conf.high), linewidth = 0.6, color = "grey50"
    ) +
    geom_point(size = 2.5, color = "grey30") +
    labs(x = "Coefficient of AI (95% CI)", y = y_label)
  ggsave(fig_path, p, width = 7, height = 4, device = png, type = "cairo")
  message("Saved ", fig_path)
}

message("Fitting topic subgroups (Figure 3) ...")
res_topic <- fit_subgroups(df, "topic")
write_csv(res_topic, file.path(out_dir, "hte_topic_R.csv"))
plot_subgroups(res_topic, "Tweet topic", file.path(out_dir, "hte_topic.png"))

message("Fitting modality subgroups (Figure 2) ...")
res_modality <- fit_subgroups(df, "multimodal")
write_csv(res_modality, file.path(out_dir, "hte_modality_R.csv"))
plot_subgroups(
  res_modality, "Tweet modality", file.path(out_dir, "hte_modality.png")
)

# Both-note subset (Table A7, second column; no figures)
df_both <- df %>% filter(tweetId %in% tid_both)

message("Fitting topic subgroups, both-note subset ...")
write_csv(
  fit_subgroups(df_both, "topic"),
  file.path(out_dir, "hte_topic_both_R.csv")
)

message("Fitting modality subgroups, both-note subset ...")
write_csv(
  fit_subgroups(df_both, "multimodal"),
  file.path(out_dir, "hte_modality_both_R.csv")
)


# ---------------------------------------------------------------------------
# robustness: Model 1 with time since note creation
# ---------------------------------------------------------------------------

ratings_df <- ratings_df %>%
  left_join(
    all_notes %>% select(noteId, noteCreatedAtMillis = createdAtMillis),
    by = "noteId"
  ) %>%
  mutate(
    time_since_note_hours = pmax(
      (createdAtMillis - noteCreatedAtMillis) / 3.6e6, 0
    ),
    log_time_since_note = log1p(time_since_note_hours),
    log_time_since_note_c =
      log_time_since_note -
      mean(log_time_since_note, na.rm = TRUE)
  )

message("Fitting m1_time_x (interacted with AI) ...")
m1_time_x <- lmer(
  update(form_main, . ~ . + AI * log_time_since_note_c),
  data = ratings_df, control = ctrl
)
save_tidy(m1_time_x, "m1_time_x")


# ---------------------------------------------------------------------------
# robustness: sensitivity to length, citations, readability, toxicity and
# valence
# ---------------------------------------------------------------------------


feats <- read_csv(
  file.path(data_dir, "note_text_features.csv"),
  col_types = cols(noteId = col_character(), tweetId = col_character())
) %>%
  drop_na(n_words, n_urls) %>%
  mutate(
    n_words_z = as.numeric(scale(n_words)),
    n_urls_z = as.numeric(scale(n_urls))
  )

toxicity <- read_csv(
  file.path(data_dir, "note_toxicity.csv"),
  col_types = cols(noteId = col_character())
) %>%
  select(noteId, toxicity)

feats_style <- feats %>%
  select(noteId, n_words, n_urls, flesch_kincaid_grade, vader_compound) %>%
  inner_join(toxicity, by = "noteId") %>%
  drop_na(n_words, n_urls, flesch_kincaid_grade, vader_compound, toxicity) %>%
  mutate(
    n_words_z = as.numeric(scale(n_words)),
    n_urls_z = as.numeric(scale(n_urls)),
    flesch_kincaid_grade_z = as.numeric(scale(flesch_kincaid_grade)),
    vader_compound_z = as.numeric(scale(vader_compound)),
    toxicity_z = as.numeric(scale(toxicity))
  )
d_style <- ratings_df %>% inner_join(feats_style, by = "noteId")


message(
  "Fitting m1_style (length, citations, readability, toxicity, valence) ..."
)
m1_style <- lmer(
  update(
    form_main,
    . ~ . + n_words_z + n_urls_z + flesch_kincaid_grade_z + toxicity_z + vader_compound_z
  ),
  data = d_style, control = ctrl
)
save_tidy(m1_style, "m1_style")

message(
  "Fitting m1_style_x (length, citations, readability, toxicity, valence x AI) ..."
)
m1_style_x <- lmer(
  update(
    form_main,
    . ~ . + AI * n_words_z + AI * n_urls_z + AI * flesch_kincaid_grade_z +
      AI * toxicity_z + AI * vader_compound_z
  ),
  data = d_style, control = ctrl
)
save_tidy(m1_style_x, "m1_style_x")



citations <- read_csv(
  file.path(data_dir, "note_source_citations.csv"),
  col_types = cols(noteId = col_character(), tweetId = col_character())
)

domain_cols <- setdiff(names(citations), c("noteId", "tweetId", "writer"))
d_sources <- ratings_df %>%
  inner_join(citations %>% select(noteId, all_of(domain_cols)), by = "noteId")%>%
  inner_join(feats_style %>%select(noteId, n_urls_z), by="noteId")



# Drop domains too rare in either writer group for a stable AI interaction
# (note-level citation counts, matching note_source_citation_features.py).
rare_domains <- domain_cols[sapply(domain_cols, function(col) {
  n_bot <- sum(citations[[col]][citations$writer == "bot"])
  n_human <- sum(citations[[col]][citations$writer == "human"])
  n_bot < 15 || n_human < 15
})]
x_domain_cols <- setdiff(domain_cols, rare_domains)

message("Fitting m1_sources (source-citation main effects) ...")
m1_sources <- lmer(
  update(
    form_main,
    as.formula(paste(". ~ . + n_urls_z +", paste(domain_cols, collapse = " + ")))
  ),
  data = d_sources, control = ctrl
)
save_tidy(m1_sources, "m1_sources")
  
message("Fitting m1_sources_x (source-citation x AI interactions) ...")
m1_sources_x <- lmer(
  update(
    form_main,
    as.formula(paste(
      ". ~ . + n_urls_z +",
      paste(sprintf("AI * %s", x_domain_cols), collapse = " + ")
    ))
  ),
  data = d_sources, control = ctrl
)
save_tidy(m1_sources_x, "m1_sources_x")


message("Done.")
