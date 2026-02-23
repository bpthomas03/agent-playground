
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# requires: scikit-learn

def load_topic_model(path):
    try:
        with open(path, "rb") as f:
            model_bundle = pickle.load(f)
        return model_bundle
    except Exception as e:
        print(f"Error loading topic model from {path}: {e}", file=sys.stderr)
        sys.exit(1)

def load_corpus_df(path):
    try:
        df = pd.read_pickle(path)
        return df
    except Exception as e:
        print(f"Error loading corpus DataFrame from {path}: {e}", file=sys.stderr)
        sys.exit(1)

def get_top_terms_per_topic(lda_model, feature_names, n_top_terms=10):
    topics_terms = []
    if hasattr(lda_model, 'components_'):
        for topic_idx, topic in enumerate(lda_model.components_):
            top_indices = topic.argsort()[::-1][:n_top_terms]
            top_terms = [feature_names[i] for i in top_indices]
            topics_terms.append(top_terms)
        return topics_terms
    else:
        print(f"Model does not have 'components_' attribute, cannot get top terms.", file=sys.stderr)
        sys.exit(1)

def get_representative_docs(lda_model, doc_term_matrix, df, n_docs=3):
    # Get topic distribution for each document
    try:
        doc_topic_dist = lda_model.transform(doc_term_matrix)
    except Exception as e:
        print(f"Error computing topic distributions: {e}", file=sys.stderr)
        sys.exit(1)
    rep_docs = []
    for topic_idx in range(doc_topic_dist.shape[1]):
        topic_values = doc_topic_dist[:, topic_idx]
        top_doc_indices = topic_values.argsort()[::-1][:n_docs]
        docs = df.iloc[top_doc_indices]
        rep_docs.append(docs)
    return rep_docs

def manual_label_suggestions(top_terms, rep_docs):
    labels = []
    for i, terms in enumerate(top_terms):
        label = ', '.join(terms[:3]).capitalize()
        labels.append(label)
    return labels

def main():
    topic_model_path = "/app/outputs/run_2026-02-23T15-44-25Z/step_3_topic_model.pkl"
    corpus_df_path = "/app/outputs/run_2026-02-23T15-44-25Z/step_2_processed_corpus_df.pkl"
    output_csv_path = "/app/outputs/run_2026-02-23T15-44-25Z/topics_labeled.csv"

    # Load model bundle (dict of lda_model, vectorizer, feature_names, etc)
    model_bundle = load_topic_model(topic_model_path)
    if not (isinstance(model_bundle, dict) and 'lda_model' in model_bundle and 'vectorizer' in model_bundle and 'feature_names' in model_bundle):
        print("Topic model file did not contain expected bundle with 'lda_model', 'vectorizer', and 'feature_names'.", file=sys.stderr)
        sys.exit(1)
    lda_model = model_bundle['lda_model']
    vectorizer = model_bundle['vectorizer']
    feature_names = model_bundle['feature_names']

    df = load_corpus_df(corpus_df_path)

    # Use processed text column for doc-term matrix
    text_col = None
    for cand in ['processed_narrative', 'processed_text', 'text', 'narrative', 'narrative_text']:
        if cand in df.columns:
            text_col = cand
            break
    if text_col is None:
        print("Could not find a processed text column in the corpus DataFrame.", file=sys.stderr)
        sys.exit(1)
    text_series = df[text_col].fillna("").astype(str)

    # Transform processed docs with saved vectorizer
    try:
        doc_term_matrix = vectorizer.transform(text_series)
    except Exception as e:
        print(f"Error transforming corpus to doc-term matrix: {e}", file=sys.stderr)
        sys.exit(1)

    n_topics = lda_model.components_.shape[0]
    n_top_terms = 10
    n_rep_docs = 3

    # Get top terms and example docs
    topics_terms = get_top_terms_per_topic(lda_model, feature_names, n_top_terms=n_top_terms)
    rep_docs = get_representative_docs(lda_model, doc_term_matrix, df, n_docs=n_rep_docs)
    topic_labels = manual_label_suggestions(topics_terms, rep_docs)

    data = []
    for i, (label, terms, docs) in enumerate(zip(topic_labels, topics_terms, rep_docs)):
        # Use 'narrative' field for example text if available, else use processed text
        example_text_col = 'narrative' if 'narrative' in docs.columns else text_col
        row = {
            "topic_id": i,
            "label": label,
            "top_terms": ", ".join(terms),
            "example_narratives": "; ".join(
                [str(d[example_text_col])[:200] for _, d in docs.iterrows()]
            )
        }
        data.append(row)
    topics_df = pd.DataFrame(data)
    try:
        topics_df.to_csv(output_csv_path, index=False)
    except Exception as e:
        print(f"Error writing CSV to {output_csv_path}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote labeled topics table to {output_csv_path}")

if __name__ == "__main__":
    main()