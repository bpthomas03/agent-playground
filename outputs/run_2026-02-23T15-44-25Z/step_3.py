
# requires: scikit-learn
import sys
import os
import pickle

import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Input and Output paths
INPUT_DF_PATH = "/app/outputs/run_2026-02-23T15-44-25Z/step_2_processed_corpus_df.pkl"

OUTPUT_MODEL_PATH = "/app/outputs/run_2026-02-23T15-44-25Z/step_3_topic_model.pkl"
OUTPUT_DISTRIBUTIONS_PATH = "/app/outputs/run_2026-02-23T15-44-25Z/step_3_document_topic_distributions.pkl"

def main():
    # Parameters for LDA
    N_TOPICS = 10  # Can be tuned or made configurable

    # Read processed DataFrame
    try:
        df = pd.read_pickle(INPUT_DF_PATH)
    except Exception as e:
        print(f"Error reading processed corpus DataFrame: {e}", file=sys.stderr)
        sys.exit(1)

    # Look for proper processed text column as written by step 2
    text_column_candidates = ['processed_narrative', 'processed_text', 'text', 'narrative', 'narrative_text']
    text_col = None
    for cand in text_column_candidates:
        if cand in df.columns:
            text_col = cand
            break
    if text_col is None:
        print("No text column found in input DataFrame. Expected one of: " + ", ".join(text_column_candidates), file=sys.stderr)
        sys.exit(1)

    # Use the found column for topic modeling
    texts = df[text_col].fillna("").astype(str).tolist()
    if len(texts) == 0:
        print("Input corpus contains no texts.", file=sys.stderr)
        sys.exit(1)

    # Vectorize the texts
    try:
        vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=1,  # Keep at 1 in case corpus is small after "__empty__" filtering
            stop_words='english'
        )
        doc_term_matrix = vectorizer.fit_transform(texts)
    except Exception as e:
        print(f"Error during vectorization: {e}", file=sys.stderr)
        sys.exit(1)

    if doc_term_matrix.shape[0] == 0 or doc_term_matrix.shape[1] == 0:
        print("Document-term matrix is empty after vectorization.", file=sys.stderr)
        sys.exit(1)

    # Fit LDA topic model
    try:
        lda = LatentDirichletAllocation(
            n_components=N_TOPICS,
            max_iter=20,
            learning_method='online',
            random_state=42,
            n_jobs=-1
        )
        lda.fit(doc_term_matrix)
    except Exception as e:
        print(f"Error during LDA fitting: {e}", file=sys.stderr)
        sys.exit(1)

    # Compute document-topic distributions (topic weights per document)
    try:
        doc_topic_distributions = lda.transform(doc_term_matrix)
    except Exception as e:
        print(f"Error computing document-topic distributions: {e}", file=sys.stderr)
        sys.exit(1)

    # Save the LDA model, vectorizer, and related data together for future analysis
    try:
        model_bundle = {
            'lda_model': lda,
            'vectorizer': vectorizer,
            'n_topics': N_TOPICS,
            'feature_names': vectorizer.get_feature_names_out()
        }
        with open(OUTPUT_MODEL_PATH, 'wb') as f:
            pickle.dump(model_bundle, f)
    except Exception as e:
        print(f"Error saving topic model: {e}", file=sys.stderr)
        sys.exit(1)

    # Save the document-topic distributions as a DataFrame, preserving index alignment
    try:
        doc_topic_df = pd.DataFrame(
            doc_topic_distributions,
            index=df.index,
            columns=[f"topic_{i}" for i in range(N_TOPICS)]
        )
        doc_topic_df.to_pickle(OUTPUT_DISTRIBUTIONS_PATH)
    except Exception as e:
        print(f"Error saving document-topic distributions: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote topic model to {OUTPUT_MODEL_PATH} and document-topic distributions to {OUTPUT_DISTRIBUTIONS_PATH}.")

if __name__ == "__main__":
    main()