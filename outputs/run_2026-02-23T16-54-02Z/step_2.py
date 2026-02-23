
# requires: nltk
import sys
import os
import pickle
import pandas as pd

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

# Download necessary NLTK data if not present (do not raise error if already downloaded)
def ensure_nltk_resource(resource_path, download_name=None):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name or resource_path.split('/')[-1])

ensure_nltk_resource('corpora/stopwords', 'stopwords')
ensure_nltk_resource('tokenizers/punkt', 'punkt')
ensure_nltk_resource('corpora/wordnet', 'wordnet')
ensure_nltk_resource('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')

import string

INPUT_PATH = "/app/outputs/run_2026-02-23T16-54-02Z/step_1_corpus_df.pkl"
OUTPUT_PATH = "/app/outputs/run_2026-02-23T16-54-02Z/step_2_processed_corpus_df.pkl"

def get_wordnet_pos(treebank_tag):
    """Map POS tag to the WordNet format for lemmatization."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text, stop_words, lemmatizer):
    try:
        # Lowercase
        text = str(text).lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords and non-alpha tokens
        tokens = [tok for tok in tokens if tok.isalpha() and tok not in stop_words]
        # POS tagging
        pos_tags = nltk.pos_tag(tokens)
        # Lemmatize
        lemmatized = [lemmatizer.lemmatize(tok, pos=get_wordnet_pos(pos)) for tok, pos in pos_tags]
        return lemmatized
    except Exception as e:
        print(f"Error during text preprocessing: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    try:
        if not os.path.exists(INPUT_PATH):
            print(f"Input file {INPUT_PATH} does not exist.", file=sys.stderr)
            sys.exit(1)
        try:
            df = pd.read_pickle(INPUT_PATH)
        except Exception as e:
            print(f"Failed to load input DataFrame: {e}", file=sys.stderr)
            sys.exit(1)

        if 'narrative' not in df.columns:
            print("Input DataFrame missing required column 'narrative'.", file=sys.stderr)
            sys.exit(1)

        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        # Process each narrative
        try:
            df['processed_text'] = df['narrative'].apply(lambda x: preprocess_text(x, stop_words, lemmatizer))
        except Exception as e:
            print(f"Error during text preprocessing: {e}", file=sys.stderr)
            sys.exit(1)

        # Save dataframe to output with processed_text column
        try:
            df.to_pickle(OUTPUT_PATH)
        except Exception as e:
            print(f"Failed to write output DataFrame: {e}", file=sys.stderr)
            sys.exit(1)

        print(f"Wrote processed corpus DataFrame to {OUTPUT_PATH}")

    except Exception as ex:
        print(f"Unexpected error: {ex}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()