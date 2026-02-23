--- Plan ---

Source ideation: /app/ideation/latest.json
Timestamp: 2026-02-23T15:22:34Z

Agreed question:
How do major recurring themes and their emotional tones in dreams change across the lifespan within a single long-running diarist, and how do these patterns compare to those in other subjects?

Steps:
  1. [load_corpus] Load the full dream corpus from data/clean_dreams.csv into a DataFrame for analysis.
      inputs:  ['data/clean_dreams.csv']
      outputs: ['corpus_df']

  2. [preprocess_text] Clean and tokenize the narrative column, removing stopwords and performing lemmatization, storing processed text for modeling.
      inputs:  ['corpus_df']
      outputs: ['processed_corpus_df']

  3. [run_lda_topic_model] Fit an LDA topic model on the processed narrative texts across the whole corpus to extract major recurring dream themes.
      inputs:  ['processed_corpus_df']
      outputs: ['topic_model', 'document_topic_distributions']

  4. [label_topics] Assign human-interpretable labels to each topic by inspecting top terms and representative narratives for each topic.
      inputs:  ['topic_model', 'processed_corpus_df']
      outputs: ['topics_labeled.csv']

  5. [assign_topic_scores] For each dream, assign the most probable topic(s) and store each dream's topic scores alongside its metadata.
      inputs:  ['document_topic_distributions', 'corpus_df']
      outputs: ['dreams_with_topics.csv']

  6. [sentiment_analysis] Use an affect/sentiment lexicon to compute the overall emotional tone for each dream narrative.
      inputs:  ['processed_corpus_df']
      outputs: ['dreams_with_sentiment.csv']

  7. [merge_topics_sentiment] Merge topic assignments and sentiment scores into a single DataFrame indexed by dream_id.
      inputs:  ['dreams_with_topics.csv', 'dreams_with_sentiment.csv']
      outputs: ['dreams_with_topics_and_sentiment.csv']

  8. [define_life_periods_barb_sanders] Define life periods or decades for Barb Sanders based on dream dates for longitudinal analysis.
      inputs:  ['dreams_with_topics_and_sentiment.csv']
      outputs: ['barb_sanders_life_periods.csv']

  9. [aggregate_barb_theme_affect_by_period] Aggregate theme prevalence and mean emotional tone per period for Barb Sanders' dreams.
      inputs:  ['dreams_with_topics_and_sentiment.csv', 'barb_sanders_life_periods.csv']
      outputs: ['barb_theme_affect_by_period.csv']

  10. [define_life_stages_others] Assign rough age groups or life stages to other subjects as data allow (e.g., based on dates).
      inputs:  ['dreams_with_topics_and_sentiment.csv']
      outputs: ['other_subjects_life_stages.csv']

  11. [aggregate_others_theme_affect_by_stage] Aggregate theme prevalence and mean emotional tone by life stage/age group for all non-Barb Sanders subjects.
      inputs:  ['dreams_with_topics_and_sentiment.csv', 'other_subjects_life_stages.csv']
      outputs: ['others_theme_affect_by_stage.csv']

  12. [compare_trajectories] Statistically compare and visualize Barb Sanders’ theme/affect trajectories to those of other subjects, highlighting universal and idiosyncratic patterns.
      inputs:  ['barb_theme_affect_by_period.csv', 'others_theme_affect_by_stage.csv', 'topics_labeled.csv']
      outputs: ['outputs/comparison_stats.json', 'outputs/comparison_plots.png']

  13. [export_key_outputs] Export all key result tables and visualizations to the outputs/ directory for further interpretation and reporting.
      inputs:  ['topics_labeled.csv', 'barb_theme_affect_by_period.csv', 'others_theme_affect_by_stage.csv', 'outputs/comparison_stats.json', 'outputs/comparison_plots.png']
      outputs: ['outputs/topics_labeled.csv', 'outputs/barb_theme_affect_by_period.csv', 'outputs/others_theme_affect_by_stage.csv', 'outputs/comparison_stats.json', 'outputs/comparison_plots.png']


---
Saved at 2026-02-23T15:22:34Z UTC.
