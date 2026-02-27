--- Plan ---

Source ideation: /app/ideation/latest.json
Timestamp: 2026-02-27T14:34:39Z

Agreed question:
How do major recurring themes and their emotional tones in dreams change across the lifespan within a single long-running diarist, and how do these patterns compare to those in other subjects?

Steps:
  1. [load_and_clean_corpus] Load the dream corpus CSV and ensure all essential columns (dream_id, subject, date, narrative, word_count) are present and datatypes are standardized; handle missing dates and unify any alternate naming or encoding of subjects.
      inputs:  ['data/clean_dreams.csv']
      outputs: ['corpus_df']
      output_spec: DataFrame with columns: dream_id (str), subject (str), date (datetime or NaT if missing/ambiguous), narrative (str), word_count (int); rows with missing narrative are dropped.
      notes: For rows with missing or ambiguous 'date' (e.g., 'Barb Sanders: #0006 (1963-??-??)'), parse what is available and set others to NaT; ensure 'narrative' is non-empty or drop the row.

  2. [preprocess_narrative_text] Process the 'narrative' column in corpus_df: lowercase, remove punctuation and stopwords, lemmatize, and create a new column 'processed_narrative'.
      inputs:  ['corpus_df']
      outputs: ['preprocessed_corpus_df']
      output_spec: DataFrame with all columns from corpus_df plus processed_narrative (str); no row may have empty processed_narrative—if preprocessing removes all tokens for a row, fallback to original narrative or the word 'empty'.
      notes: If all tokens are removed during preprocessing, retain original text or insert a placeholder ('empty') to ensure no document is empty for downstream vectorization. All downstream steps should use preprocessed_narrative for modeling.

  3. [split_by_group_and_life_stage] Assign each dream in preprocessed_corpus_df to (a) 'Barb Sanders' or 'others', and (b) a life stage (e.g., by decade or custom period) based on date or, if missing, by best-guess from metadata.
      inputs:  ['preprocessed_corpus_df']
      outputs: ['grouped_corpus_df']
      output_spec: DataFrame with all columns from preprocessed_corpus_df plus columns: group ('Barb Sanders' or 'others'), life_stage (string: e.g., '1960s', '1970s', or binned e.g. 'age_20s' if age data exists, otherwise decade from date).
      notes: For missing dates, infer life_stage from best available info in subject/dream_id; if undeterminable, label as 'unknown'. This enables trajectory aggregation by life stage.

  4. [vectorize_corpus] Create a document-term matrix for topic modeling using the processed_narrative column of grouped_corpus_df.
      inputs:  ['grouped_corpus_df']
      outputs: ['doc_term_matrix', 'vectorizer']
      output_spec: doc_term_matrix: sparse matrix (documents x terms); vectorizer: fitted sklearn-like object with get_feature_names_out(). The mapping between dream_id and row index must be preserved.
      notes: Do not apply min_df/max_df parameters so stringency does not drop all tokens for any document. Order must match grouped_corpus_df.

  5. [run_topic_model] Perform LDA topic modeling on the document-term matrix to identify major themes across the whole corpus.
      inputs:  ['doc_term_matrix', 'vectorizer']
      outputs: ['lda_model']
      output_spec: Trained LDA model object (e.g., sklearn.decomposition.LatentDirichletAllocation) compatible with the vectorizer used; the model is not serialized to disk yet.
      notes: Choose number of topics via coherence or fixed value (with rationale). Retain model and vectorizer together for extraction of top terms and topic distributions.

  6. [extract_topic_distributions] Infer topic probabilities (distributions) for each dream and save as a DataFrame indexed by dream_id.
      inputs:  ['lda_model', 'doc_term_matrix', 'grouped_corpus_df']
      outputs: ['topic_distributions_df']
      output_spec: DataFrame with columns: dream_id, subject, group, life_stage, topic_0, topic_1, ..., topic_{N-1}, where each topic_k is the per-dream probability for topic k.
      notes: dream_id must match grouped_corpus_df order; one topic probability vector per row; down-the-line aggregation expects this column layout.

  7. [label_topics_with_top_terms] Extract the top N words for each topic from the LDA model and vectorizer, and assign each topic an interpretable label or summary of top terms.
      inputs:  ['lda_model', 'vectorizer']
      outputs: ['topic_labels_df']
      output_spec: DataFrame with columns: topic_id (int), top_terms (comma-separated str, e.g., 'water, boat, river, drown, swim'), and optionally theme_label (if manual labeling occurs).
      notes: Top terms to be human interpretable; if no manual label, set theme_label = top_terms.

  8. [compute_affect_scores] Apply an affect or sentiment lexicon to each dream's processed_narrative to estimate emotional valence (and, if possible, arousal or other affect dimensions).
      inputs:  ['grouped_corpus_df']
      outputs: ['affect_scores_df']
      output_spec: DataFrame with columns: dream_id, subject, group, life_stage, valence_score (float), [optional: arousal_score (float), affect_label (str)]; one row per dream.
      notes: If processed_narrative has no scorable words, set valence_score to neutral (e.g., 0) and flag in affect_label.

  9. [aggregate_theme_and_affect_trajectories] Aggregate (a) mean topic prevalence (per theme), and (b) mean affect scores, for each group (Barb Sanders vs. others) and life stage.
      inputs:  ['topic_distributions_df', 'affect_scores_df']
      outputs: ['theme_trajectories_df', 'affect_trajectories_df']
      output_spec: theme_trajectories_df: DataFrame with columns: life_stage, group, topic_id, mean_prevalence (float), dream_count. affect_trajectories_df: DataFrame with columns: life_stage, group, metric (valence/arousal/affect_label), value (float or str), dream_count.
      notes: If a life stage has too few dreams (e.g., less than N=5), flag or skip in outputs, to prevent noisy or misleading averages.

  10. [statistical_comparison] For each theme and affect metric, statistically compare trajectories between Barb Sanders and others across life stages (e.g., t-tests, effect sizes).
      inputs:  ['theme_trajectories_df', 'affect_trajectories_df']
      outputs: ['statistical_comparison_results.csv']
      output_spec: CSV file with columns: life_stage, topic_id (or metric), group_comparison, test_statistic, p_value, effect_size, interpretation.
      notes: Adjust p-values for multiple comparisons. If some life stages have insufficient data, denote NA for those cells.

  11. [visualize_and_export_results] Visualize and export dream theme and affect trajectories, with interpretable labels on plots and axes; output all trajectories and plots as CSVs and images for reporting.
      inputs:  ['theme_trajectories_df', 'affect_trajectories_df', 'topic_labels_df', 'statistical_comparison_results.csv']
      outputs: ['outputs/theme_trajectories.csv', 'outputs/affect_trajectories.csv', 'outputs/statistical_comparison_results.csv', 'outputs/theme_trajectories_plot.png', 'outputs/affect_trajectories_plot.png']
      output_spec: theme_trajectories.csv: columns life_stage, group, topic_id, topic_label (from topic_labels_df, top_terms or theme_label), mean_prevalence, dream_count. affect_trajectories.csv: columns life_stage, group, metric, value, dream_count. Plots: PNGs with life_stage on x-axis, mean theme prevalence/affect values on y-axis, separate lines for groups, with themes labeled using topic_labels_df.
      notes: Do not use generic 'theme1' etc.; all plots and CSVs must use human-interpretable labels from topic_labels_df. Ensure all listed files are written for downstream reporting and manuscript use.


---
Saved at 2026-02-27T14:34:39Z UTC.
