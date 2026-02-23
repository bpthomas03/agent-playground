
# requires: nltk
import sys
import os
import pickle
import pandas as pd

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except ImportError:
    print("Required package nltk is not installed.", file=sys.stderr)
    sys.exit(1)

INPUT_PATH = '/app/outputs/run_2026-02-23T15-44-25Z/step_1_corpus_df.pkl'
OUTPUT_PATH = '/app/outputs/run_2026-02-23T15-44-25Z/step_2_processed_corpus_df.pkl'

def ensure_nltk_resources():
    """Downloads necessary NLTK resources if not already present."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

def preprocess_text(text, stop_words, lemmatizer):
    # Basic cleaning: lowercasing, tokenization, stopword removal, lemmatization
    if not isinstance(text, str) or not text.strip():
        return ""
    try:
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t.isalpha()]  # Keep only words
        tokens = [t for t in tokens if t not in stop_words]
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        if len(tokens) == 0:
            # If after processing nothing remains, attempt fallback: keep minimal info
            ascii_letters = [c for c in text if c.isalpha()]
            if len(ascii_letters) > 0:
                # Return a single string with joined ascii letters (fallback)
                return ''.join(ascii_letters)
            else:
                # Cannot salvage; mark as empty
                return "__empty__"
        return ' '.join(tokens)
    except Exception as e:
        # On unexpected error, fallback to raw non-punctuation letters as above
        ascii_letters = [c for c in text if c.isalpha()]
        if len(ascii_letters) > 0:
            return ''.join(ascii_letters)
        return "__empty__"

def main():
    # Make sure required NLTK data is available
    try:
        ensure_nltk_resources()
    except Exception as e:
        print(f"Failed to get NLTK resources: {str(e)}", file=sys.stderr)
        sys.exit(1)

    # Load input corpus DataFrame
    try:
        df = pd.read_pickle(INPUT_PATH)
    except FileNotFoundError:
        print(f"Input file not found: {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Failed to read input file: {str(e)}", file=sys.stderr)
        sys.exit(1)

    # Validate that 'narrative' column exists
    if 'narrative' not in df.columns:
        print("'narrative' column not found in input DataFrame.", file=sys.stderr)
        sys.exit(1)

    # Prepare stopwords and lemmatizer
    try:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
    except Exception as e:
        print(f"Failed to initialize nltk tools: {str(e)}", file=sys.stderr)
        sys.exit(1)

    # Preprocess narratives
    try:
        df['processed_narrative'] = df['narrative'].apply(
            lambda x: preprocess_text(x, stop_words, lemmatizer)
        )
    except Exception as e:
        print(f"Error during text preprocessing: {str(e)}", file=sys.stderr)
        sys.exit(1)

    # Only keep rows where processed_narrative is NOT "__empty__" and not empty string
    nonempty_df = df[(df['processed_narrative'].str.strip() != "__empty__") & (df['processed_narrative'].str.strip() != "")]
    removed = len(df) - len(nonempty_df)

    if len(nonempty_df) == 0:
        print("ERROR: All documents are empty after preprocessing. Cannot proceed to topic modeling.", file=sys.stderr)
        sys.exit(1)
    else:
        df = nonempty_df

    if len(df) == 0:
        print("All documents are empty in the DataFrame. Please check the input data.", file=sys.stderr)
        sys.exit(1)

    try:
        df.to_pickle(OUTPUT_PATH)
    except Exception as e:
        print(f"Could not write output file: {str(e)}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote processed corpus DataFrame with lemmatized, stopword-removed narratives to {OUTPUT_PATH}.")

if __name__ == '__main__':
    main()